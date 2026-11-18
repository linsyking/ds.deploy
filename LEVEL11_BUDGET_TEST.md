# Maxwell level 11: thinking-budget comparison

The new benchmark task is `maxwell-1-11` in the adjacent `ai-sde-benchmark`
repository. It uses canonical `tasks/maxwell/run/levels/1-11.txt`, the same
instructions, engine, verifier, category (`maxwell`), difficulty (0.9), and
one-hour timeout as level 10. The compiled engine is an ignored build artifact,
copied from the existing identical engine used by levels 6–10.

Task discovery, preconditions, initial unsolved state, and rejection of an empty
move string passed. The task is automatically included in the Maxwell category.
It can be selected independently with:

```sh
python -m bench run -t maxwell-1-11 -a claude-ds -j 1
```

That command selects the task with the ordinary client configuration. It does
not itself supply any numeric thinking budget.

## Settings and execution

All measured arms used the local SGLang deployment, production DSpark block 5
with overlap enabled, DeepSeek V4.1 Flash, Claude Code 2.1.317, **Max effort**,
64,000 total output tokens per response, and `--autocompact 500k`. MTI was off.
The only intended per-arm setting change was the explicit thinking-token cap.

Private loopback relays supplied `thinking.type=enabled` and the selected
`budget_tokens`; the client continued to send `output_config.effort=max`.
This adapter is needed because positive `MAX_THINKING_TOKENS` values did not
produce a numeric budget for this DeepSeek alias in the installed Claude Code.
Global user settings and production defaults were not changed. Relay metadata
contains limits, usage, stop reasons and retokenized thinking lengths, without
adding request text, thinking text or authentication headers to its logs.

The user requested concurrent execution. The initial sequential partial trial
was excluded, and four fresh arms (16K, 24K, 32K, 60K) started together at
18:18:26 on November 3, 2026, local time. The user subsequently stopped the
unfinished 24K and 60K trials and requested 12K, 8K and 4K. Those three fresh
arms started together at 18:48:54. Each had the normal one-hour task timeout;
none of the completed trials timed out.

## Results

| Thinking cap | Score | Time | Reported output tokens | Retokenized thinking tokens | Responses near cap | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 2,617.8 s / 43.6 min | 1,199,039 | 1,127,254 | 256 | 2 |
| 8,192 | 0 | 1,969.1 s / 32.8 min | 862,878 | 836,626 | 85 | 1 |
| 12,288 | 0 | 1,534.4 s / 25.6 min | 670,613 | 653,936 | 45 | 1 |
| 16,384, earlier four-arm batch | **100** | **989.7 s / 16.5 min** | **406,092** | 398,907 | 22 | 0 |
| 32,768, earlier four-arm batch | 0 | 924.2 s / 15.4 min | 382,111 | 377,563 | 7 | 0 |
| 24,576 | unfinished, stopped by user | — | incomplete | incomplete | — | at least 1 |
| 60,000 | unfinished, stopped by user | — | incomplete | incomplete | — | at least 1 |

All scores above come from the canonical game verifier. The 16K solution ended
in `State: accept`. Every completed unsuccessful submission ended in
`State: none`; all agents exited normally, without infrastructure errors or
timeouts. The stopped trials have no final accuracy result and are not failures.
No completed arm hit the 64,000-token total-output ceiling.

Reported output includes thinking, visible output, and compaction requests.
Thinking lengths are counted by retokenizing streamed thinking text with the
deployment tokenizer, rather than a separate server-reported reasoning-usage
field. “Near cap” means within 16 tokens of the configured thinking cap.

The 4K run attempted three Python commands to parse board/level data, violating
the instruction to use only the game interface and no scripts. Its answer still
failed verification. The other completed arms' visible logs contained no Python,
compiler or network-tool keywords in the heuristic audit. That audit is not a
formal proof of full instruction compliance. All saved workdirs contained only
the original game, instructions, level file and `solutions.txt`; the three input
files were byte-identical to the task originals.

## Interpretation

**16K remains the best observed successful configuration.** In this level-11
trial, lowering the thinking cap to 12K, 8K or 4K did not improve correctness,
elapsed time or total output. Smaller caps produced more thinking/tool rounds;
a shorter individual thinking block did not imply a shorter agent task.
The failed 4K run spent almost three times the successful 16K run's output tokens.

This is one unseeded trial per budget on one puzzle. It does not establish a
monotonic relationship or a global optimum; the unsuccessful 32K trial also
shows that a larger cap is not sufficient for success. Sampling and native
numerical variation can affect the reasoning path.

Timing is under shared concurrent load. The first batch started with four arms,
the second with three, and contention declined as arms finished. Comparisons
between batches are therefore not measurements of isolated decoder latency.
Production prefix-cache state and other shared-service traffic are additional
uncontrolled factors. No weighted combination of correctness and time is imposed.

The 4K arm compacted twice, taking 10.808 and 9.128 seconds. The 8K and 12K arms
compacted once, taking 14.517 and 12.223 seconds. First compactions reduced
roughly 469K–475K estimated tokens to 9K–15K. Claude reserves context headroom,
so the nominal 500K window acts below 500K retained tokens. These observations
do not independently identify an optimal compaction window.

## Artifacts and integrity

- Three smaller-budget arms:
  `.runs/level11-budget/20261103-184854-parallel-small/`.
- Consolidated results, including the earlier completed references:
  `.runs/level11-budget/20261103-184854-parallel-small/summary.json`.
- Earlier four-arm batch:
  `.runs/level11-budget/20261103-181826-parallel/`, including
  `STOPPED_BY_USER.json` for 24K/60K.
- Excluded sequential setup:
  `.runs/level11-budget/20261103-181019/EXCLUDED.json`.
- Each arm's private native benchmark results are under its `results/` directory;
  `result.json` records the exact run path.
- `artifact-audit.json`, `input-integrity.json`, and `cleanup.json` in the
  smaller-budget directory record validation and cleanup.

Final checks found the production API healthy, every temporary test process
stopped, and production configuration, global Claude settings, benchmark config,
and level-11 task inputs unchanged throughout the comparison. `serving.json`
still matches Git HEAD and the saved production configuration byte for byte.
No production default was changed.

A subsequent DSH headless trial with 16K thinking and Max effort is recorded in
`DSH_BENCHMARK.md`. It scored 0 in 471.3 seconds, exiting normally before the
one-hour timeout, with 255,000 output tokens and no compactions. The earlier
Claude 16K result above remains the successful reference; different concurrency
and native harness behavior limit direct latency comparisons.
