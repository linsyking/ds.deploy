# Local thinking-budget and compaction screen

Status: the approximately 30-minute screen is complete. Results below are
provisional single trials, not an established optimum.

The subsequent [level-11 comparison](LEVEL11_BUDGET_TEST.md) tested 4K, 8K, 12K,
16K and 32K with Max effort. Only 16K solved that puzzle in the completed trials;
24K and 60K were stopped by the user before completion. The later comparison
used concurrent budget arms and has different timing limitations from this screen.

Production DSpark remains enabled with block size 5 and overlap scheduling.
MTI is disabled. The model, service configuration, global Claude settings,
benchmark sources, task inputs and evaluators are unchanged.

## Method

All new arms use the local Anthropic endpoint, Claude Code 2.1.317, the same
DeepSeek model with `[1m]`, effort Max, all five Maxwell tasks, and `-j 5`.
Total output remains 64,000 tokens per response while the explicit thinking
cap is varied. The last arm uses the fastest fully completed budget setting
with a smaller context-compaction window. Arms run sequentially.

Private client settings and benchmark configs isolate each arm. An owned
loopback relay supplies explicit `thinking.type=enabled` and `budget_tokens`
on thinking requests whose output limit can accommodate that budget. It logs
only limit fields, response usage, stop reasons and retokenized thinking lengths.
No request text, thinking text or authentication headers are added to these logs.
Existing native Claude session records supply compaction markers and per-task usage.

The overall screen is time-bounded. An arm stopped to preserve the screen's time
allowance is censored: unfinished tasks are not counted as incorrect, and that
arm cannot establish final suite accuracy or completion time. Output accounting
on interrupted requests can be incomplete. Full arms use the maximum native task
duration as their suite completion time; launcher polling adds up to 15 seconds
to the recorded `elapsed_wall_s`.

## Results

| Setting | Verified result | Suite completion / observation | Reported output tokens | Compactions |
| --- | --- | ---: | ---: | ---: |
| Earlier production: no explicit thinking cap, 64K output, 500K compaction | 5/5 | 1,425 s | 1,576,410 | 0 |
| 16,384 thinking, 64K output, 500K compaction | 5/5 | 766.2 s | 868,677 | 0 |
| 32,768 thinking, 64K output, 500K compaction | 3 correct, 2 unfinished | censored at 510.1 s | 793,318, incomplete | 0 |
| 16,384 thinking, 64K output, 128K compaction | 2 correct, 3 unfinished | censored at 510.1 s | 809,970, incomplete | 7 |

The completed 16K arm took approximately 46% less time and generated 45% fewer
output tokens than the earlier production trial. The earlier run is a historical
reference, not a repeated contemporaneous control. Unseeded sampling, native
BF16 sparse-attention ties, prefix-cache state and the shared production service
limit causal conclusions. Repeated trials are needed before claiming a stable gain.

The 16K arm reached its thinking cap on 33 responses, had a largest response of
16,608 tokens, and had zero total-output truncations. All five verifiers accepted
the solutions. Level 10 took 766.2 seconds and reached 351,439 input tokens;
the 500K compaction window never fired. Per-task details and counters are in the
artifacts below.

The 32K arm completed levels 6, 8 and 9 correctly in 318.2, 459.8 and 193.6
seconds. Levels 7 and 10 were unfinished when its allocation ended. Its largest
response was 32,915 tokens; a 32K total-output limit would have cut that response.
It had 13 thinking-cap hits and no total-output truncations. These partial
results do not establish that 32K is less accurate or slower to complete the suite.

The 128K arm completed levels 8 and 9 correctly in 248.2 and 313.2 seconds;
levels 6, 7 and 10 were unfinished at the screen deadline. Seven completed
compactions took 15.9–48.9 seconds each, totaling 181.3 seconds across concurrent
tasks. This sum is not extra suite wall time. The first compaction of each task
reduced roughly 98K–112K pre-compaction tokens to 12K–22K. Claude reserves
headroom, so the nominal 128K window compacts below 128K retained tokens.
Levels 6 and 7 compacted twice; the other levels compacted once. There were 37
thinking-cap hits, no total-output truncations, and a largest response of 22,418
tokens. The screen did not establish a completion-time or final-quality advantage
for the smaller window.

The 16K arm completed within its 15-minute allowance; the remaining screen time
was divided between the 32K arm and the compaction arm, giving each 8.5 minutes.
Their shorter observation periods are a material limitation: fewer verified
solutions at those cutoffs cannot be interpreted as lower final accuracy or a
longer full-suite completion time than the 16K arm.

## Configuration implications

The fastest fully verified provisional setting is 16,384 thinking tokens,
64,000 total output tokens, and a 500K compaction window. A lower total-output
ceiling does not directly accelerate a response that already ends below it.
Keeping 64K leaves headroom without requiring the model to generate that many
tokens. A 32K ceiling would have accommodated all responses in the completed
16K trial, but it has not been independently benchmarked here.

`max_tokens` covers thinking plus the visible answer or tool call for one API
response. The thinking allowance resets for subsequent responses. Compaction
acts on accumulated conversation context and adds summarization work; it is
separate from ending a thinking block at its budget.

Keep 500K provisionally because it has a complete successful result and avoids
summarization on these measured trajectories. The screen does not identify the
optimal compaction window: 128K was censored, and intermediate windows such as
256K were not tested. Likewise, it does not establish 16K as globally optimal
against a fully completed 32K comparison.

Claude Code documents `MAX_THINKING_TOKENS` for fixed thinking budgets and
`CLAUDE_CODE_MAX_OUTPUT_TOKENS` for total output. However, actual local wire tests
of this DeepSeek alias still sent adaptive thinking with no numeric budget when
positive `MAX_THINKING_TOKENS` was set. Adding
`CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1`, disabling discovery, or declaring custom
capabilities did not change that result. The installed disable-adaptive branch
only applies to Opus/Sonnet 4.6. Do not assume exporting the budget variable alone
applies this recommendation to the current DeepSeek alias; the experiment uses
an explicit request adapter. The output variable and `--autocompact` flag did
apply in these experiments.

Official references: [environment variables](https://code.claude.com/docs/en/env-vars)
and [model/compaction configuration](https://code.claude.com/docs/en/model-config).

## Related observations

Previous level-10 production, ordinary-decoding and MTI runs had zero Claude Code
compactions, with peak inputs of 257,785, 283,295 and 127,911 tokens respectively.
Production did hit its 64,000-token total-output ceiling eight times. Truncation
and context compaction are distinct. Cumulative generated output is not the
amount retained in context at once.

The 16K DSpark arm used almost the same total output as the previous ordinary MTI
trial (868,677 versus 872,170), with a much shorter observed wall time (766.2 versus
3,242 seconds). Neither comparison establishes separate thinking-token reduction
in the old MTI trial. Combining MTI with DSpark remains a plausible engineering
experiment, described in [MTI_DSPARK_DESIGN.md](MTI_DSPARK_DESIGN.md), and has not
been implemented or tested.

## Artifacts

- Sweep: `.runs/budget-sweep/20261103-173318/`, including per-arm manifests,
  `result.json`, `requests.json`, private CLI sessions, and input-integrity hashes.
- Native completed 16K results: `../ai-sde-benchmark/results/20261103-173319/`.
- Native censored 32K results: `../ai-sde-benchmark/results/20261103-174619/`.
- Native censored 128K results: `../ai-sde-benchmark/results/20261103-175449/`.
- CLI environment wire tests: `.runs/endpoint-budget/20261103-172945-cli-env/summary.json`.
- Previous level-10 compaction audit: `.runs/endpoint-budget/level10-compaction.json`.
- Historical production/MTI comparison: `MTI_EXPERIMENT.md`.

Final checks found all temporary test clients and relays stopped, the production
API healthy, and input-integrity hashes unchanged. `serving.json` still matches
Git HEAD and the saved production configuration byte for byte. No production
defaults were changed. Consolidated metrics are in the sweep's `summary.json`;
cleanup evidence is in `cleanup.json`.
