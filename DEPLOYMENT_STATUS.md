# Deployment status — 2026-11-12

## PDF document input on c02

`documents.py` adds Anthropic document support before native request validation.
Both user attachments and documents inside `tool_result.content` expand into
page-aligned extracted text and rendered images. The installed SGLang validator
and multimodal converter then receive their existing text/image block types.
No installed SGLang source files are edited. Ordinary form fields are included
in rendering; scanned pages retain their image even when text extraction is empty.

Base64 PDF, PDF URLs, and UTF-8/plain text sources are supported. Files API IDs
and native citation events are explicitly unsupported. Rendering uses CPU
PDFium 5.15.2 at 144 DPI with bounded page size, a bounded ten-minute memory
cache, and serialized native access off the API event loop. Full local limits
and installation commands are in README.md.

`enable_documents=true` is the only added serving setting. Heretic's 38-tensor
overlay, DSpark block 5, vision, thinking defaults/budgets, endpoint, bearer
credential and the FRPC/GOST route to server port 8312 are retained. Exa query
extraction runs before expansion so document text does not become search input.

Validation: 23 document adapter tests, 14 existing search tests, one native
SGLang validator/converter regression (both attachment placements), and eight
existing checkpoint/swap tests passed. An initial live check passed all six
API cases, and Claude Code Read correctly read the text code, image-only code,
and chart. Final checks after the middleware-order restart passed: all six
PDF API cases, Claude Code Read (including its nested base64 PDF document),
and all eight standard text/vision/tool/SSE checks. The final service is healthy
with PID 2449408. Evidence: `audit/documents-20261112-083619/` and
`audit/validation-20261112-083623/`; configuration/overlay comparison is saved
in `.runs/documents/final.json`.

SGLang's `/v1/messages/count_tokens` accepts normalized document requests but
retains the native estimate, which undercounts vision tokens. Generation
`usage.input_tokens` includes the actual image expansion. This is not full
Anthropic token-counting or citation parity.

## Exa web search

On 2026-11-01 the user requested adding Exa and restarting the service.
The provided Exa credential is in .secrets/exa-api-key, mode 0600, and is not
part of Git or normal runtime sync. A direct Exa Auto request succeeded with
three results and highlights; provider-reported cost was $0.007.

web_search.py adds basic Anthropic /v1/messages search compatibility using
SGLang's installed ExaClient, with five results, domain filters, a ten-minute
bounded cache, and coalescing for concurrent duplicate searches. It supplies
search evidence to the model rather than implementing Anthropic server-tool
events or dynamic filtering. Ordinary client tools and images are preserved;
requests without server search tools pass through unchanged. Provider failure
returns an explicit 502. The key and query text are not logged by this adapter.

enable_web_search=true is set in serving.json; ds41ctl loads EXA_API_KEY and
registers the adapter only in the API process, without editing package files.
Both API and Exa credentials are redacted from controller output. Existing
API URLs, authentication, and forwarding remain in use.

Validation before restart: all 14 no-network adapter tests passed, Python
compilation and git diff --check passed, and middleware registration succeeded
inside the deployment's CUDA/SGLang environment. Restart was launched; live
search and normal API regression checks are pending readiness.

## c02 migration

The checkpoint and SGLang environment have been migrated from c04 to c02
(10.78.202.30). All 83 checkpoint files (510,311,045,295 bytes) were hash-verified
on c02 at 18:32 PDT on 2026-10-31. The engine became ready at 18:45:40 PDT.
Live checks confirmed four GB200 GPUs with Recovery Action None, API
health, bearer authentication, and OpenAI/Anthropic text inference. The API key
matches c04. c02 uses uv-managed Python 3.11.15 and GCC/G++ 13.
Full c02 vision, reasoning, and explicit-budget validation remains pending.
The older c04 measurements below are historical, not c02 validation results.
No c13 changes were made.

## Running service

- c04 (10.78.202.32), four NVIDIA GB200 GPUs; no c13 changes in this deployment.
- Model: deepseek-ai/DeepSeek-V4.1-Flash-1017, revision 1bae0e1088993b9b82de08f59de5be5a34e42170.
- Native checkpoint weights and quantization; no additional weight re-quantization.
- SGLang 0.5.23, PyTorch 2.14.1, FlashInfer 0.7.2 / cu130, NVIDIA driver 580.167.08.
- TP4 / EP4, native dsv4 attention and flashinfer_mxfp4 MoE backend.
- Vision enabled; native OpenAI and Anthropic APIs; automatic reasoning/tool parsers.
- DSpark speculative decoding enabled, block size 5.
- Reasoning defaults updated to thinking ON / max (100); explicit request overrides remain supported. This is separate from speculative decoding.
- FP8 E4M3 KV cache, 1,048,576-token context, bounded decoder SWA replay.
- Selected prefill chunk: 16,384. Static memory fraction: 0.80. Max running requests: 16. Decode graph max batch: 32.
- Scratch-local CUDA/JIT caches; service managed by ds41ctl and tmux.

## Thinking configuration update

The Anthropic endpoint previously treated output_config.effort="max" without an
explicit thinking toggle as non-thinking. Controlled non-streaming and streaming
requests reproduced zero reasoning in that case. Explicit thinking.type="adaptive"
with max produced reasoning blocks. The launcher now supplies native defaults
{"thinking": true, "reasoning_effort": "max"}; live verification passed nine reasoning checks. Returned prompt IDs decode to
Reasoning Effort: 100 and an open thinking header; explicit low decodes to 50.
Explicit off still produces no reasoning. Vision and tool round trips also work
with the new default, including streamed thinking. See
audit/reasoning-check-20261031-154801/ for detailed evidence and the noted
random-image probe limitation.
Claude Code files are unchanged.

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

Benchmark conditions: non-thinking pre-tokenized synthetic input IDs through /v1/completions,
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

- FRPC 0.71.1 runs on c02 through GOST 3.3.0 at 127.0.0.1:18080, with transport TLS enabled.
- Server: ssh.linsyking.com:17000; server port 8312 forwards directly to c02 10.78.202.30:18000.
- The tracked old c04 FRPC process was stopped to release port 8312. c04 is not an intermediate hop.
- c02 FRPC login and proxy registration succeeded at 18:55:47 PDT on 2026-10-31.
- Server-local OpenAI base URL: http://127.0.0.1:8312/v1; Anthropic base URL: http://127.0.0.1:8312.
- API key: unchanged, in .secrets/api-key. Use Authorization: Bearer <key>.
- FRP credentials are preserved in .secrets/frpc.toml; the route now targets c02.
- FRPC was verified against the official release SHA-256 checksum; installation evidence is in .runs/frpc-install/install.json.
- Use ./ds41ctl tunnel-start, tunnel-status, and tunnel-stop. Startup after a machine reboot is manual.
- Server-local access is the intended scope; no firewall rules were changed.

## Files and evidence

- Production entrypoints: ds41ctl and sglang_load.py; tunables in serving.json.
- audit/deployment-validation.json: live settings, versions, authentication and GPU checks.
- audit/validation-20261031-150510/: eight native API/vision/streaming checks.
- audit/performance/: request-level timings, summaries, and comparison.json.
- audit/performance/benchmark-source.tar.gz: reproducible benchmark source/configs, checksum manifest alongside.
- Obsolete comparison/download/engine scripts and old results remain only in verified audit archives.
- Model weights, installed environments, and useful caches were preserved.

See README.md for operation and client instructions.
