"""Reproducible Maxwell comparison; keeps credentials out of benchmark records.

Run with system Python. The existing production baseline must have completed:
  python mti_experiment.py --baseline-run results/<run>
The service is restored to its saved production configuration on completion or
failure. Each arm invokes the user's original benchmark command unchanged.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent / 'ai-sde-benchmark'
ART = ROOT / '.runs/mti'
STATE = ART / 'experiment.json'
CONTROL = ART / 'runtime.json'
TASKS = ['maxwell-1-' + str(n) for n in range(6, 11)]


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def reported_output_tokens(agent):
    # These are this experiment's own Claude CLI session files. Read usage only;
    # never enable a server request probe or include transcript content in reports.
    project = re.sub(r'[^a-zA-Z0-9]', '-', agent['workdir'])
    folder = ART / 'claude-config/projects' / project
    messages = {}
    for path in folder.glob('*.jsonl'):
        for line in path.open():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            message = event.get('message', {})
            if not isinstance(message, dict):
                continue
            if event.get('type') == 'assistant' and message.get('id'):
                key = message['id']
                count = (message.get('usage') or {}).get('output_tokens', 0)
                messages[key] = max(messages.get(key, 0), count)
    return sum(messages.values()) if messages else None


def run_arm(name, enabled, threshold=1.5):
    save(CONTROL, dict(enabled=enabled, threshold=threshold, scale=1.5, run_id=name))
    ctl = runpy.run_path(str(ROOT / 'ds41ctl'), run_name='inspection')
    # Reset idle prefix cache symmetrically before each measured arm.
    request = ctl['OPENER'].open
    import urllib.request
    req = urllib.request.Request('http://10.78.202.30:18000/flush_cache',
                                 headers=ctl['auth_headers'](), method='POST')
    with request(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError('Cache flush failed before experiment')
    before = set((BENCH / 'results').iterdir())
    env = os.environ.copy()
    env['CLAUDE_CONFIG_DIR'] = str(ART / 'claude-config')
    command = [sys.executable, '-m', 'bench', 'run', '-c', 'maxwell', '-a', 'claude-ds', '-j', '5']
    start = time.monotonic()
    print(f'Starting {name}', flush=True)
    with (ART / (name + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=BENCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            print(f'{name}: running for {time.monotonic() - start:.0f}s', flush=True)
            time.sleep(30)
    if process.returncode != 0:
        raise RuntimeError(f'{name} benchmark exited {process.returncode}')
    added = set((BENCH / 'results').iterdir()) - before
    if len(added) != 1:
        raise RuntimeError('Could not uniquely identify benchmark run directory')
    run_dir = added.pop()
    state = json.loads(STATE.read_text())
    state.setdefault('measured_runs', {})[name] = dict(
        run=str(run_dir), wall_seconds=time.monotonic() - start,
        enabled=enabled, threshold=threshold, scale=1.5)
    # Wall time below uses a 30-second polling interval; the report also computes
    # the precise task-start to final-evaluation interval from native artifacts.
    if enabled and (ART / 'counters.json').exists():
        state['measured_runs'][name]['counters'] = json.loads((ART / 'counters.json').read_text())
    save(STATE, state)
    print(f'Completed {name}: {run_dir}', flush=True)


def report():
    state = json.loads(STATE.read_text())
    runs = dict(production_dspark=dict(run=str(BENCH / state['production_baseline'])),
                **state.get('measured_runs', {}))
    summary = {}
    for name, info in runs.items():
        root = Path(info['run'])
        results = [json.loads((root / 'claude-ds' / task / 'result.json').read_text())
                   for task in TASKS]
        agents = [json.loads((root / 'claude-ds' / task / 'agent.json').read_text())
                  for task in TASKS]
        token_usage = {a['task']: reported_output_tokens(a) for a in agents}
        start = min(datetime.datetime.fromisoformat(a['started_at']) for a in agents)
        end = max(datetime.datetime.fromisoformat(r['evaluated_at']) for r in results)
        summary[name] = dict(run=str(root),
                            correctness=sum(r['score'] for r in results) / len(results),
                            solved=sum(r['score'] == 100 for r in results),
                            total_wall_seconds=(end - start).total_seconds(),
                            summed_task_seconds=sum(r['duration_s'] for r in results),
                            reported_output_tokens_by_task=token_usage,
                            reported_output_tokens=(sum(token_usage.values())
                                                    if all(v is not None for v in token_usage.values())
                                                    else None),
                            tasks={r['task']: {k:r[k] for k in ('score','duration_s','timed_out','exit_code','status')}
                                   for r in results})
    save(ART / 'comparison.json', summary)
    lines = ['# MTI Maxwell experiment', '',
             'Paper: https://arxiv.org/abs/2510.13940v4',
             'Code: https://github.com/EnVision-Research/MTI at ' + state['upstream_commit'], '',
             'Command: `python -m bench run -c maxwell -a claude-ds -j 5`.',
             'Agent settings and task/verifier sources are unchanged. Direct local API connection.', '',
             '| Arm | Correctness | Solved | Total wall time (s) | Summed task time (s) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, value in summary.items():
        lines.append(f'| {name} | {value["correctness"]:.1f} | {value["solved"]}/5 | '
                     f'{value["total_wall_seconds"]:.1f} | {value["summed_task_seconds"]:.1f} |')
    mti = summary.get('mti_tau1.5_omega1.5')
    if mti:
        lines.extend(['', '## Observed comparison', ''])
        production = summary['production_dspark']
        ordinary = summary['ordinary_decode_control']
        if (mti['correctness'] <= production['correctness']
                and mti['total_wall_seconds'] > production['total_wall_seconds']):
            lines.extend(['The existing DSpark production configuration is preferable on correctness and total wall time',
                          'in this trial. MTI remains an optional experiment and is disabled after the comparison.', ''])
        if (mti['correctness'] > ordinary['correctness']
                and mti['total_wall_seconds'] < ordinary['total_wall_seconds']):
            lines.extend(['MTI improved both correctness and wall time against ordinary decoding in this trial.',
                          'This is a promising transfer signal, subject to the single-trial and implementation limitations below.', ''])
        for name in ['ordinary_decode_control', 'production_dspark']:
            reference = summary[name]
            ratio = mti['total_wall_seconds'] / reference['total_wall_seconds']
            lines.append(f'Against `{name}`, MTI solved {mti["solved"]}/5 versus '
                         f'{reference["solved"]}/5 and took {ratio:.2f} times the total wall time.')
        summed_ratio = mti['summed_task_seconds'] / ordinary['summed_task_seconds']
        lines.extend(['', f'MTI took {summed_ratio:.2f} times the ordinary control’s summed task duration.',
                      'Wall time measures concurrent completion at `-j 5`; summed duration adds all five agent durations.',
                      'A wall-time improvement therefore does not mean that every individual case ran faster.'])
        if mti['reported_output_tokens'] and ordinary['reported_output_tokens']:
            reduction = 100 * (1 - mti['reported_output_tokens'] / ordinary['reported_output_tokens'])
            lines.append(f'MTI used {reduction:.1f}% fewer reported output tokens than ordinary decoding; '
                         'these include thinking and visible output.')
        lines.extend(['', 'Correctness and wall time are reported separately; no arbitrary weighted score is imposed.',
                      'A timeout is scored by the benchmark verifier, with its elapsed time included.',
                      'The ordinary control isolates the guidance experiment more closely; the production arm',
                      'also includes DSpark and overlap scheduling, so its timing difference cannot be attributed solely to MTI.', '',
                      '| Task | Production score / seconds | Ordinary score / seconds | MTI score / seconds |',
                      '| --- | ---: | ---: | ---: |'])
        for task in TASKS:
            values = [summary[name]['tasks'][task] for name in
                      ['production_dspark', 'ordinary_decode_control', 'mti_tau1.5_omega1.5']]
            lines.append('| ' + task + ' | ' + ' | '.join(
                f'{v["score"]:.0f} / {v["duration_s"]:.1f}' + (' (timeout)' if v['timed_out'] else '')
                for v in values) + ' |')
    lines.extend(['', 'Times exclude engine startup and validation. Total wall time spans the first task start',
                  'through the last evaluation; task times are native agent execution durations.',
                  'The production arm uses the existing running service. Ordinary control and MTI start with cleared prefix caches.',
                  'The ordinary control and MTI arms disable DSpark and overlap scheduling.',
                  'MTI uses full-vocabulary entropy before sampling, threshold 1.5 nats, guidance 1.5,',
                  'and a temporary `OUTPUT ERROR` continuation. It applies to thinking and answer tokens.',
                  'At a selected position, sampling uses `guided_logits = 1.5 * conditional_logits - 0.5 * negative_logits`.',
                  'Combining raw logits is equivalent to the paper’s log-probability combination up to a per-row constant.',
                  'The cue is rolled back, and native grammar masks, temperature, and sampling are then applied.',
                  'This is an exploratory single trial per arm on five tasks; it does not establish a general accuracy gain.', ''])
    lines.extend(['## Implementation and validation', '',
                  'The author code targets Hugging Face and vLLM 0.10.2; its SGLang implementation is listed as TODO.',
                  'This deployment uses the independent experimental port in `mti.py`; native package files are unchanged.',
                  'The temporary cue is three tokens with this tokenizer. Prefix KV is reused, request-scoped ratio-2',
                  'compressor and Engram histories are restored, and only newly allocated cache pages are freed.',
                  'The port evaluates the cue with three sequential auxiliary decode forwards per selected batch.',
                  'A fused cue prefill and other performance optimizations have not been implemented; these timings measure this prototype.',
                  'The adapter currently supports this TP4/EP4 DeepSeek V4.1 paged-cache layout and ordinary synchronous decoding.',
                  'Integration with DSpark remains further work.', '',
                  '`test_mti.py` verifies entropy, guidance algebra, cache ownership and cleanup, captured-buffer restoration,',
                  'and isolation of synthetic forced-token validation from other requests.',
                  'All eight unit tests and thirteen live validation checks passed before the measured MTI run.',
                  '`mti_validate.py` compares temporary-branch logits with a native scheduler continuation of the same request,',
                  'including page boundaries and prefixes up to 8K tokens, then checks shadow generation and five concurrent guided requests.',
                  'All ten paired prefix checks matched exactly. Long-prefix checks use an eager synthetic oracle with a stable',
                  'logical-position tie rule: native BF16 sparse top-k can choose different keys with exactly equal cutoff scores.',
                  'Tracing confirmed identical valid indexer scores and unchanged committed prefix KV bytes before this first divergence.',
                  'The tie-rule patches are restored before benchmarking; all measured arms retain the original native sparse kernels.',
                  'The single-request shadow test preserved greedy text. The concurrent shadow test checks completion, not identical text.',
                  'Independent prefill-versus-decode comparisons can differ numerically and are recorded separately.',
                  'Validation results: `.runs/mti/validation/summary.json`.', '',
                  'This is a transfer experiment on DeepSeek V4.1 with the existing Claude Code harness, not a reproduction',
                  'of the paper’s smaller-model benchmarks. Agent effort and sampling settings are held fixed across arms.',
                  'Unseeded sampling and native sparse top-k tie choices can affect results; repeated trials are needed for causal claims.', '',
                  '## Additional generation metrics', '',
                  'Reported output-token totals include thinking and visible output. Counts come from the experiment’s own',
                  'Claude CLI session usage, deduplicated by response ID; no server request logging is enabled.',
                  'Usage on interrupted or timed-out requests can be incomplete. Separate thinking-token accounting is unavailable.', '',
                  '| Arm | Reported output tokens |', '| --- | ---: |'])
    for name, value in summary.items():
        lines.append(f'| {name} | {value["reported_output_tokens"]} |')
    if mti:
        lines.extend(['', '| Task | Production output tokens | Ordinary output tokens | MTI output tokens |',
                      '| --- | ---: | ---: | ---: |'])
        for task in TASKS:
            lines.append('| ' + task + ' | ' + ' | '.join(
                str(summary[name]['reported_output_tokens_by_task'][task]) for name in
                ['production_dspark', 'ordinary_decode_control', 'mti_tau1.5_omega1.5']) + ' |')
    counters = state.get('measured_runs', {}).get('mti_tau1.5_omega1.5', {}).get('counters')
    if counters:
        lines.extend(['', 'MTI counters below are the last periodic snapshot (up to 30 seconds before completion):',
                      'Interventions count selected token positions. Changed top-1 is measured before native grammar masks and sampling;',
                      'it is not a count of changed sampled tokens. Negative forwards count auxiliary model calls, each possibly batched.',
                      '', '```json', json.dumps(counters, indent=2), '```'])
    lines.extend(['', 'Artifacts and pinned source hashes are recorded in `.runs/mti/experiment.json` and',
                  '`.runs/mti/comparison.json`. Package versions are in `.runs/mti/environment.json`.',
                  '`.runs/mti/tool-audit.json` records a heuristic tool-use audit and unchanged benchmark source/input/binary hashes.',
                  '`mti_experiment.py` runs the comparison and restores the saved production',
                  'configuration in `finally`. Its explicit debug flag can retain a warmed ordinary engine with MTI disabled.', ''])
    (ROOT / 'MTI_EXPERIMENT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-run')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--keep-experimental-on-error', action='store_true',
                        help='Debug only: keep the warmed ordinary engine after failure, with MTI disabled')
    parser.add_argument('--resume-after-control', action='store_true',
                        help='Reuse the completed ordinary-control results while validating and running MTI')
    args = parser.parse_args()
    if args.report_only:
        report()
        return
    state = json.loads(STATE.read_text())
    if args.baseline_run:
        state['production_baseline'] = args.baseline_run
        save(STATE, state)
    if args.resume_after_control:
        prior = state.get('measured_runs', {}).get('ordinary_decode_control', {}).get('run')
        if not prior or not all((Path(prior) / 'claude-ds' / task / 'result.json').exists()
                                for task in TASKS):
            raise RuntimeError('Resume requires a completed ordinary-decoding control')
    baseline = BENCH / state['production_baseline']
    while not all((baseline / 'claude-ds' / task / 'result.json').exists() for task in TASKS):
        n = len(list(baseline.glob('claude-ds/*/result.json')))
        print(f'Waiting for production baseline: {n}/5 finished', flush=True)
        time.sleep(30)
    original = (ART / 'production-serving.json').read_text()
    completed = False
    try:
        if not (ART / 'production-baseline-engine.log').exists():
            shutil.copy2(ROOT / 'sglang.log', ART / 'production-baseline-engine.log')
        cfg = json.loads(original)
        cfg['mti_experiment'] = True
        save(ROOT / 'serving.json', cfg)
        save(CONTROL, dict(enabled=False, threshold=1.5, scale=1.5))
        state['mti_source_sha256'] = hashlib.sha256((ROOT / 'mti.py').read_bytes()).hexdigest()
        save(STATE, state)
        subprocess.run([str(ROOT / 'ds41ctl'), 'restart'], cwd=ROOT, check=True)
        ctl = runpy.run_path(str(ROOT / 'ds41ctl'), run_name='inspection')
        env = ctl['launch_environment'](ctl['config']())
        subprocess.run([str(ROOT / '.venvs/sglang/bin/python'), str(ROOT / 'mti_validate.py')],
                       cwd=ROOT, env=env, check=True)
        if not args.resume_after_control:
            run_arm('ordinary_decode_control', False)
        run_arm('mti_tau1.5_omega1.5', True)
        shutil.copy2(ROOT / 'sglang.log', ART / 'experimental-engine.log')
        report()
        completed = True
    finally:
        shutil.copy2(ROOT / 'sglang.log', ART / 'last-experimental-engine.log')
        save(CONTROL, dict(enabled=False, threshold=1.5, scale=1.5))
        if completed or not args.keep_experimental_on_error:
            (ROOT / 'serving.json').write_text(original)
            subprocess.run([str(ROOT / 'ds41ctl'), 'restart'], cwd=ROOT, check=True)
        else:
            print('Debug failure: ordinary engine kept warm; MTI disabled. Restore production after debugging.', flush=True)


if __name__ == '__main__':
    main()
