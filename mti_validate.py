"""Live synthetic cache and logits validation for the experimental SGLang port."""
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / '.runs/mti/runtime.json'
DEST = ROOT / '.runs/mti/validation'
CFG = json.loads((ROOT / 'serving.json').read_text())
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def control(**kwargs):
    tmp = CONTROL.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict({'enabled': False, 'threshold': 1.5, 'scale': 1.5}, **kwargs)))
    tmp.replace(CONTROL)


def flush_cache():
    # Independent requests must start from the same cache state. Otherwise the
    # scheduler may replace freshly prefetched pages with an existing radix
    # entry after the initial sample. That entry can come from a different
    # quantized prefill shape (e.g. prefix+cue versus prefix alone).
    request = urllib.request.Request(
        f'http://{CFG["host"]}:{CFG["port"]}/flush_cache', method='POST',
        headers={'Authorization': 'Bearer ' + (ROOT / '.secrets/api-key').read_text().strip()})
    with OPENER.open(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError('Could not normalize validation cache state')


def generate(ids, length=1, rid=None):
    # Input logprobs force full prefill instead of the deployment's bounded
    # decoder replay, making the independent reference numerically comparable.
    body = dict(input_ids=ids, return_logprob=True, logprob_start_len=0,
                sampling_params=dict(temperature=0, max_new_tokens=length,
                                                   ignore_eos=True))
    if rid:
        body['rid'] = rid
    request = urllib.request.Request(
        f'http://{CFG["host"]}:{CFG["port"]}/generate',
        data=json.dumps(body).encode(), headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + (ROOT / '.secrets/api-key').read_text().strip(),
        })
    with OPENER.open(request, timeout=180) as response:
        return json.load(response)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CFG['model_path'], trust_remote_code=False)
    cue = tokenizer.encode('OUTPUT ERROR', add_special_tokens=False)
    prefix = tokenizer.encode('Consider this arithmetic carefully. 19 times 23 is', add_special_tokens=False)
    filler = tokenizer.encode(' The known value is 17.', add_special_tokens=False)
    varied = tokenizer.encode('\n'.join(
        f'Record {n}: left value {n * 61 % 997}, right value {n * 23 % 283}; '
        f'category {n % 17}, previous {n * n % 709}.' for n in range(1000)),
        add_special_tokens=False)
    checks = []
    try:
        for size in [127, 128, 129, 255, 256, 257, 1023, 4095, 4096, 8193]:
            ids = ([0] + filler * (size // len(filler) + 10))[:size - len(prefix)] + prefix
            if size >= 1023:
                ids = ([0] + varied)[:size - len(prefix)] + prefix
            label = f'{int(time.time()*1000)}-{size}'
            control(capture='prefill-' + label, validation_rid='prefill-' + label)
            flush_cache()
            generate(ids + cue, rid='prefill-' + label)
            control(capture='ref-' + label, validation_force_ids=cue,
                    validation_rid='ref-' + label, validation_paired_branch=True,
                    validation_sync=False, validation_disable_graph=size >= 1023,
                    validation_stable_topk=size >= 1023)
            flush_cache()
            generate(ids, len(cue) + 1, rid='ref-' + label)
            control(capture='repeat-' + label, validation_force_ids=cue, validation_rid='repeat-' + label)
            flush_cache()
            generate(ids, len(cue) + 1, rid='repeat-' + label)
            control(enabled=True, threshold=0., scale=1.5, shadow=True,
                    capture='branch-' + label, validation_rid='branch-' + label)
            flush_cache()
            generate(ids, rid='branch-' + label)
            reference = torch.load(DEST / ('ref-' + label + '.pt'), weights_only=True)['raw'][0]
            repeated = torch.load(DEST / ('repeat-' + label + '.pt'), weights_only=True)['raw'][0]
            paired = torch.load(DEST / ('ref-' + label + '-initial.pt'), weights_only=True)
            initial = paired['raw'][0]
            prefill = torch.load(DEST / ('prefill-' + label + '.pt'), weights_only=True)['raw'][0]
            branch = torch.load(DEST / ('branch-' + label + '.pt'), weights_only=True)
            actual = paired['negative'][0]
            delta = actual.float() - reference.float()
            centered = delta - delta.mean()
            item = dict(test=f'negative_logits_prefix_{size}',
                        oracle=('stable_sparse_topk_eager' if size >= 1023 else 'native_graph'),
                        same_top1=int(actual.argmax()) == int(reference.argmax()),
                        centered_rmse=float(centered.square().mean().sqrt()),
                        max_centered_error=float(centered.abs().max()),
                        initial_prefix_rmse=float((initial - branch['raw'][0]).square().mean().sqrt()),
                        temporary_branch_repeat_rmse=float((actual - paired['negative_repeat'][0]).square().mean().sqrt()),
                        native_repeat_rmse=float((reference - repeated).square().mean().sqrt()),
                        native_prefill_decode_rmse=float(((reference - prefill) - (reference - prefill).mean()).square().mean().sqrt()))
            # Quantized prefill and decode may differ slightly. A cache/position
            # error must not be excused as a small numerical discrepancy.
            item['passed'] = (item['same_top1'] and item['centered_rmse'] < .25
                              and item['temporary_branch_repeat_rmse'] < .25)
            checks.append(item)
            print(json.dumps(item), flush=True)
        control()
        flush_cache()
        normal = generate(prefix, 64)
        control(enabled=True, threshold=0., scale=1.5, shadow=True)
        flush_cache()
        shadow = generate(prefix, 64)
        checks.append(dict(test='shadow_generation_preserves_greedy_text',
                           passed=normal['text'] == shadow['text']))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            responses = list(pool.map(lambda _: generate(prefix, 32), range(5)))
        checks.append(dict(test='five_parallel_shadow_requests',
                           identical_text=all(r['text'] == responses[0]['text'] for r in responses),
                           passed=all(r['meta_info']['completion_tokens'] == 32 for r in responses)))
        control(enabled=True, threshold=0., scale=1.5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            responses = list(pool.map(lambda n: generate(prefix + filler * n, 32), range(5)))
        checks.append(dict(test='five_parallel_guided_requests',
                           passed=all(r['meta_info']['completion_tokens'] == 32 and r['text']
                                      for r in responses)))
        (DEST / 'summary.json').write_text(json.dumps(checks, indent=2) + '\n')
        if not all(c['passed'] for c in checks):
            raise RuntimeError('MTI live validation failed; inspect .runs/mti/validation/summary.json')
        print('All MTI live validation checks passed', flush=True)
    finally:
        control()


if __name__ == '__main__':
    main()
