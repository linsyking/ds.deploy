#!/usr/bin/env python3
"""Show the configured SGLang requests, cache, throughput, and memory."""
import argparse
import datetime
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(base, path):
    key = ROOT / '.secrets/api-key'
    headers = {'Authorization': 'Bearer ' + key.read_text().strip()} if key.exists() else {}
    req = urllib.request.Request(base + path, headers=headers)
    with OPENER.open(req, timeout=10) as response:
        return response.read().decode()


def metrics(text):
    result = {}
    pools = {}
    for line in text.splitlines():
        if not line.startswith('sglang:'):
            continue
        body, _, value = line.rpartition(' ')
        try:
            number = float(value)
        except ValueError:
            continue
        name = body[7:].split('{', 1)[0]
        labels = dict(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', body))
        if labels.get('tp_rank', '0') != '0' or labels.get('pp_rank', '0') != '0':
            continue
        pool = next((labels[k] for k in ['pool', 'pool_name', 'kv_pool'] if k in labels), None)
        if pool:
            pools.setdefault(pool, {})[name] = number
        elif 'is_streaming' in labels:
            result[name] = result.get(name, 0) + number
        else:
            result[name] = number
    return result, pools


def display(value, suffix='', digits=0):
    return 'n/a' if not isinstance(value, (int, float)) else f'{value:,.{digits}f}{suffix}'


def render(base):
    values, pools = metrics(fetch(base, '/metrics'))
    info = json.loads(fetch(base, '/server_info'))
    states = info.get('internal_states') or [{}]
    memory = states[0].get('memory_usage') or {}
    cfg = info.get('server_args') or info
    maximum = info.get('effective_max_running_requests_per_dp') or cfg.get('max_running_requests')
    print(f'SGLang load — {datetime.datetime.now():%Y-%m-%d %H:%M:%S} — {base}')
    print('Requests: ' + display(values.get('num_running_reqs')) + ' running, ' +
          display(values.get('num_queue_reqs')) + ' queued; configured max ' + display(maximum))
    print('Model: ' + str(cfg.get('served_model_name', 'deepseek-v4.1-flash-1017')) +
          '; TP4; context ' + display(cfg.get('context_length')))
    print('Vision configured: ' + str(cfg.get('enable_multimodal', 'n/a')) +
          '; prefill chunk ' + display(cfg.get('chunked_prefill_size')))
    defaults = cfg.get('default_chat_template_kwargs') or {}
    print('Thinking default: ' + str(defaults.get('thinking', 'engine default')) +
          '; effort: ' + str(defaults.get('reasoning_effort', 'engine default')))
    print()
    capacity = memory.get('token_capacity')
    if any(name in values for name in ['kv_used_tokens', 'kv_evictable_tokens', 'kv_available_tokens']):
        print('KV cache (engine-reported accounting units):')
        for label, value in [('used', values.get('kv_used_tokens')),
                             ('evictable', values.get('kv_evictable_tokens')),
                             ('free', values.get('kv_available_tokens')),
                             ('capacity', capacity)]:
            print(f'  {label:12} {display(value):>14}')
    if pools:
        print('Hybrid cache pools (kept separate; compressed tokens are not context length):')
        for pool, data in sorted(pools.items()):
            print('  ' + pool + ': ' + ', '.join(f'{key}={display(value)}' for key, value in sorted(data.items())))
    if memory.get('token_capacity_swa') is not None:
        print('SWA pool capacity: ' + display(memory['token_capacity_swa']) + ' (separate pool accounting units)')
    prompt = values.get('prompt_tokens_total')
    cached = values.get('cached_tokens_total')
    if prompt is not None and cached is not None and prompt > 0:
        print(f'Prefix cache, cumulative: {cached / prompt:.1%} ({cached:,.0f}/{prompt:,.0f} prompt tokens)')
    elif 'cache_hit_rate' in values:
        print(f'Prefix cache, current reported gauge: {values["cache_hit_rate"]:.1%} (not a cumulative hit rate)')
    else:
        print('Prefix cache hit rate: n/a')
    print('Generation throughput, last reported batch: ' + display(values.get('gen_throughput'), ' tok/s', 1))
    print('Speculative acceptance length: ' + display(values.get('spec_accept_length'), ' tok/step', 2))
    print('KV usage gauge: ' + display(values.get('token_usage'), '', 3))
    if memory:
        graph = memory.get('graph') or {}
        graph_gb = sum(graph.values()) if graph else memory.get('cuda_graph')
        print('Memory per TP rank, engine-reported GB: ' +
              'weights=' + display(memory.get('weight'), digits=2) +
              ', CUDA graphs=' + display(graph_gb, digits=2) +
              ', available at startup=' + display(memory.get('startup_available'), digits=2))
        kv_gb = memory.get('kvcache')
        print('KV allocation GB: ' + (display(kv_gb, digits=2) if kv_gb else
              'not reported by this hybrid-cache engine; use pool counters above'))
    print('Note: gauges can lag during prefill; cache capacity does not guarantee concurrency at a given context.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('interval', nargs='?', type=float, default=0, help='Refresh interval in seconds; default prints once')
    parser.add_argument('--base-url', help='Override the endpoint in serving.json')
    args = parser.parse_args()
    if args.interval < 0:
        parser.error('interval cannot be negative')
    cfg = json.loads((ROOT / 'serving.json').read_text())
    base = (args.base_url or f'http://{cfg["host"]}:{cfg["port"]}').rstrip('/')
    while True:
        try:
            render(base)
        except (OSError, ValueError, KeyError) as error:
            print(f'Unable to read SGLang status: {error}')
            if not args.interval:
                return 1
        if not args.interval:
            return 0
        time.sleep(args.interval)
        print()


if __name__ == '__main__':
    raise SystemExit(main())
