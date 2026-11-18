# MTI Maxwell experiment

Paper: https://arxiv.org/abs/2510.13940v4
Code: https://github.com/EnVision-Research/MTI at ac848621b601fef1b61807513dce3f2023bb0e82

Command: `python -m bench run -c maxwell -a claude-ds -j 5`.
Agent settings and task/verifier sources are unchanged. Direct local API connection.

| Arm | Correctness | Solved | Total wall time (s) | Summed task time (s) |
| --- | ---: | ---: | ---: | ---: |
| production_dspark | 100.0 | 5/5 | 1425.0 | 3271.8 |
| ordinary_decode_control | 80.0 | 4/5 | 3600.0 | 9954.0 |
| mti_tau1.5_omega1.5 | 100.0 | 5/5 | 3242.0 | 10794.9 |

## Observed comparison

The existing DSpark production configuration is preferable on correctness and total wall time
in this trial. MTI remains an optional experiment and is disabled after the comparison.

MTI improved both correctness and wall time against ordinary decoding in this trial.
This is a promising transfer signal, subject to the single-trial and implementation limitations below.

Against `ordinary_decode_control`, MTI solved 5/5 versus 4/5 and took 0.90 times the total wall time.
Against `production_dspark`, MTI solved 5/5 versus 5/5 and took 2.28 times the total wall time.

MTI took 1.08 times the ordinary control’s summed task duration.
Wall time measures concurrent completion at `-j 5`; summed duration adds all five agent durations.
A wall-time improvement therefore does not mean that every individual case ran faster.
MTI used 47.0% fewer reported output tokens than ordinary decoding; these include thinking and visible output.

Correctness and wall time are reported separately; no arbitrary weighted score is imposed.
A timeout is scored by the benchmark verifier, with its elapsed time included.
The ordinary control isolates the guidance experiment more closely; the production arm
also includes DSpark and overlap scheduling, so its timing difference cannot be attributed solely to MTI.

| Task | Production score / seconds | Ordinary score / seconds | MTI score / seconds |
| --- | ---: | ---: | ---: |
| maxwell-1-6 | 100 / 962.4 | 0 / 3600.2 (timeout) | 100 / 2382.5 |
| maxwell-1-7 | 100 / 507.8 | 100 / 1718.4 | 100 / 1339.9 |
| maxwell-1-8 | 100 / 203.1 | 100 / 1334.9 | 100 / 1505.9 |
| maxwell-1-9 | 100 / 173.6 | 100 / 1287.1 | 100 / 2324.5 |
| maxwell-1-10 | 100 / 1424.9 | 100 / 2013.4 | 100 / 3242.1 |

Times exclude engine startup and validation. Total wall time spans the first task start
through the last evaluation; task times are native agent execution durations.
The production arm uses the existing running service. Ordinary control and MTI start with cleared prefix caches.
The ordinary control and MTI arms disable DSpark and overlap scheduling.
MTI uses full-vocabulary entropy before sampling, threshold 1.5 nats, guidance 1.5,
and a temporary `OUTPUT ERROR` continuation. It applies to thinking and answer tokens.
At a selected position, sampling uses `guided_logits = 1.5 * conditional_logits - 0.5 * negative_logits`.
Combining raw logits is equivalent to the paper’s log-probability combination up to a per-row constant.
The cue is rolled back, and native grammar masks, temperature, and sampling are then applied.
This is an exploratory single trial per arm on five tasks; it does not establish a general accuracy gain.

## Implementation and validation

The author code targets Hugging Face and vLLM 0.10.2; its SGLang implementation is listed as TODO.
This deployment uses the independent experimental port in `mti.py`; native package files are unchanged.
The temporary cue is three tokens with this tokenizer. Prefix KV is reused, request-scoped ratio-2
compressor and Engram histories are restored, and only newly allocated cache pages are freed.
The port evaluates the cue with three sequential auxiliary decode forwards per selected batch.
A fused cue prefill and other performance optimizations have not been implemented; these timings measure this prototype.
The adapter currently supports this TP4/EP4 DeepSeek V4.1 paged-cache layout and ordinary synchronous decoding.
Integration with DSpark remains further work.

`test_mti.py` verifies entropy, guidance algebra, cache ownership and cleanup, captured-buffer restoration,
and isolation of synthetic forced-token validation from other requests.
All eight unit tests and thirteen live validation checks passed before the measured MTI run.
`mti_validate.py` compares temporary-branch logits with a native scheduler continuation of the same request,
including page boundaries and prefixes up to 8K tokens, then checks shadow generation and five concurrent guided requests.
All ten paired prefix checks matched exactly. Long-prefix checks use an eager synthetic oracle with a stable
logical-position tie rule: native BF16 sparse top-k can choose different keys with exactly equal cutoff scores.
Tracing confirmed identical valid indexer scores and unchanged committed prefix KV bytes before this first divergence.
The tie-rule patches are restored before benchmarking; all measured arms retain the original native sparse kernels.
The single-request shadow test preserved greedy text. The concurrent shadow test checks completion, not identical text.
Independent prefill-versus-decode comparisons can differ numerically and are recorded separately.
Validation results: `.runs/mti/validation/summary.json`.

This is a transfer experiment on DeepSeek V4.1 with the existing Claude Code harness, not a reproduction
of the paper’s smaller-model benchmarks. Agent effort and sampling settings are held fixed across arms.
Unseeded sampling and native sparse top-k tie choices can affect results; repeated trials are needed for causal claims.

## Additional generation metrics

Reported output-token totals include thinking and visible output. Counts come from the experiment’s own
Claude CLI session usage, deduplicated by response ID; no server request logging is enabled.
Usage on interrupted or timed-out requests can be incomplete. Separate thinking-token accounting is unavailable.

| Arm | Reported output tokens |
| --- | ---: |
| production_dspark | 1576410 |
| ordinary_decode_control | 1645539 |
| mti_tau1.5_omega1.5 | 872170 |

| Task | Production output tokens | Ordinary output tokens | MTI output tokens |
| --- | ---: | ---: | ---: |
| maxwell-1-6 | 458965 | 629664 | 184468 |
| maxwell-1-7 | 221176 | 275228 | 96502 |
| maxwell-1-8 | 80413 | 209932 | 109460 |
| maxwell-1-9 | 69504 | 202373 | 178465 |
| maxwell-1-10 | 746352 | 328342 | 303275 |

MTI counters below are the last periodic snapshot (up to 30 seconds before completion):
Interventions count selected token positions. Changed top-1 is measured before native grammar masks and sampling;
it is not a count of changed sampled tokens. Negative forwards count auxiliary model calls, each possibly batched.

```json
{
  "tokens": 872096,
  "interventions": 132257,
  "changed_top1": 75081,
  "negative_forwards": 320256,
  "branch_seconds": 1385.9346584118903,
  "threshold": 1.5,
  "scale": 1.5,
  "shadow": false,
  "run_id": "mti_tau1.5_omega1.5"
}
```

Artifacts and pinned source hashes are recorded in `.runs/mti/experiment.json` and
`.runs/mti/comparison.json`. Package versions are in `.runs/mti/environment.json`.
`.runs/mti/tool-audit.json` records a heuristic tool-use audit and unchanged benchmark source/input/binary hashes.
`mti_experiment.py` runs the comparison and restores the saved production
configuration in `finally`. Its explicit debug flag can retain a warmed ordinary engine with MTI disabled.
