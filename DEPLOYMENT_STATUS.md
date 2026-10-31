# Deployment status — 2026-10-31

## Running service

- c04 (10.78.202.32), four NVIDIA GB200 GPUs; no c13 changes in this deployment.
- Model: deepseek-ai/DeepSeek-V4.1-Flash-1017, revision 1bae0e1088993b9b82de08f59de5be5a34e42170.
- Native checkpoint weights and quantization; no additional weight re-quantization.
- SGLang 0.5.23, PyTorch 2.14.1, FlashInfer 0.7.2 / cu130, NVIDIA driver 580.167.08.
- TP4 / EP4, native dsv4 attention and flashinfer_mxfp4 MoE backend.
- Vision enabled; native OpenAI and Anthropic APIs; automatic reasoning/tool parsers.
- DSpark speculative decoding enabled, block size 5.
- FP8 E4M3 KV cache, 1,048,576-token context, bounded decoder SWA replay.
- Selected prefill chunk: 16,384. Static memory fraction: 0.80. Max running requests: 16. Decode graph max batch: 32.
- Scratch-local CUDA/JIT caches; service managed by ds41ctl and tmux.

## Measured selection

16K is the best of the three tested prefill settings for this long-context workload.
This is a measured choice, not a claim of a globally optimal parameter combination.
Other parameters were fixed across the comparison; DSpark block size was not swept.

Cold first-token latency, seconds (mean of two runs; median across requests within each run):

| Input tokens | Concurrency | 4K chunk | 8K chunk | 16K chunk |
| ---: | ---: | ---: | ---: | ---: |
| 131,072 | 1 | 3.232 | 1.615 | 1.022 |
| 131,072 | 2 | 4.864 | 2.447 | 1.537 |
| 480,000 | 1 | 12.367 | 6.037 | 4.088 |
| 480,000 | 2 | 18.324 | 9.192 | 6.217 |

At 480K tokens, one-user cold TTFT improved from 12.367s to 4.088s.
The selected profile averaged about 0.127s first-token latency and 1.88ms per
output token for a warm repeated 480K prompt with one user. Warm decode results
are broadly similar across profiles and noisy at this sample size; the clear
advantage of 16K is long-prompt prefill latency.

Benchmark conditions: pre-tokenized synthetic input IDs through /v1/completions,
256 output tokens, temperature 0, one synchronized wave per point, cold-prefix
and repeated-prefix passes. Input IDs exclude text-tokenization time, so real
chat/coding and vision latency can differ. A warm pass is an attempted prefix
reuse, not a guarantee that every prefix remains cached. No SLA or sustained-load
claim is made from these short runs.

## Acceptance checks

- All 72 tuning requests passed across 48 points (three profiles, two repeats).
- All 114 final acceptance requests passed across 14 points.
- 1,000,000 input tokens tested at concurrency 1 only: cold TTFT 10.212s; repeated-prefix TTFT 0.166s.
- 100K and 200K tokens/request passed at concurrency 4, 8, and 16, cold and repeated-prefix.
- At 16 × 200K, cold median TTFT was 13.296s; the whole wave finished in 26.057s.
- 16 is the configured and tested cap, not a proven hardware maximum; 16 × 500K/1M was not tested.
- All eight text, vision, tool-call/result, and streaming Anthropic validation checks passed.
- Vision validation uses an image-only random code, preventing a text-only answer from passing.
- Final API auth checks: anonymous /v1/models returns 401; authenticated request returns 200.
- All four c04 GPUs report Recovery Action None after the stress test.

A non-fatal upstream metadata-glue graph capture warning falls back to eager
metadata preparation on each rank. Target-verification and draft CUDA graphs
remain enabled; all reported requests succeeded. No engine package source was
patched. The controller redacts credentials and preserves structured access logs.

## Server-local tunnel

- GOST 3.3.0 runs in the background on 127.0.0.1:18080, using https://1.1.1.1/dns-query.
- FRPC 0.71.1 ARM64 was downloaded from its official release and checksum-verified.
- FRPC uses transport.proxyURL=http://127.0.0.1:18080 and transport TLS.
- Server: ssh.linsyking.com:17000; remote port 8312 forwards to c04 10.78.202.32:18000.
- FRPC login and proxy registration succeeded; the background process is running.
- Intended OpenAI base URL on your server: http://127.0.0.1:8312/v1.
- Intended Anthropic base URL on your server: http://127.0.0.1:8312.
- API key: .secrets/api-key, separate from the FRP transport token.
- Credentials are ignored by Git, mode 0600, under a mode-0700 directory.
- You confirmed server-local access only. External port 8312 access is not required; no firewall/public exposure changes were made.
- A server-side loopback request was not tested from c04. Local API authentication, GOST routing, FRPC login and proxy registration were verified separately.

Use ./ds41ctl tunnel-start, tunnel-status, and tunnel-stop for FRPC.
The processes survive SSH logout; startup after a machine reboot is manual.

## Files and evidence

- Production entrypoints: ds41ctl and sglang_load.py; tunables in serving.json.
- audit/deployment-validation.json: live settings, versions, authentication and GPU checks.
- audit/tunnel-validation.json: tunnel checks and confirmed access scope.
- audit/validation-20261031-150510/: eight native API/vision/streaming checks.
- audit/performance/: request-level timings, summaries, and comparison.json.
- audit/performance/benchmark-source.tar.gz: reproducible benchmark source/configs, checksum manifest alongside.
- Obsolete comparison/download/engine scripts and old results remain only in verified audit archives.
- Model weights, installed environments, and useful caches were preserved.

See README.md for operation and client instructions.
