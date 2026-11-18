# Level-11 urgency prompt and independent solver race

Two fresh trials run sequentially against the local DSpark / DeepSeek deployment
using Claude Code 2.1.317, Max effort, 16,384 thinking tokens, 64,000 total output
tokens per response, a nominal 500K compaction window, and a one-hour task limit.
All parent and child requests receive the same inference controls. At most eight
new subagents may be spawned per trial.

1. **Math-olympiad with urgency.** The pinned, unmodified skill and common
   instructions from [the previous comparison](SKILL_BENCHMARK.md) are reused.
   The task prompt adds exactly `solve the problem as fast as possible`.
   The default launcher prompt is otherwise preserved.
2. **No-skill, eight independent solvers.** A coordinator is instructed to launch
   eight general-purpose Claude Code agents concurrently, each solving the
   entire level independently in its own fresh workdir. All eight receive the
   same puzzle and rules; no strategies or subproblems are assigned by the parent.
   The Skill and Workflow tools are disabled for this condition.
   Each solver must verify its candidate using the game CLI and save its own
   `solutions.txt`. The coordinator waits for the first completed solver reporting
   a full verified solution, copies that answer, stops the other solvers, and
   exits. It must not solve, refine, compare candidates, or wait for all solvers.
   If an agent finishes without a complete candidate, it waits for the others.
   The normal external benchmark verifier determines correctness of the answer
   actually submitted. Compliance with this topology will also be audited.

Native Claude subagents share a model service but have separate solver workdirs.
The total task time includes dispatch, waiting, cancellation and finalization.
No prior solutions or hints are supplied. Fresh private Claude profiles disable
auto-memory; task/game inputs and production/global settings are unchanged.
The existing locked cap hook bounds total spawns, including nested attempts.

The first arm started November 4, 2026 at 07:54:02 local time. Artifacts:

`.runs/skill-tests/20261104-075402-speed/` (urgency run and first race attempt).

The corrected race is in `.runs/skill-tests/20261104-083704-speed/` and started
at 08:37:04. `latest-speed.json` now points to this active race.

`manifest.json` records both conditions; `state.json` records the active trial;
`comparison.json` accumulates completed results. The runtime source is archived
under `source-code/`. `.runs/skill-tests/latest-speed.json` points to this queue;
the earlier five-condition queue and its pointer are preserved.

| Condition | Verifier score | Time | Output tokens | Subagents |
| --- | ---: | ---: | ---: | ---: |
| math-olympiad + urgency | not evaluated; canceled by user | ~1,771.3 s / 29.5 min | ≥1,865,934 | 2 |
| No-skill race, N=8 | not evaluated; canceled by user | ~1,506.3 s / 25.1 min | ≥3,918,640 | 8 |

The user stopped the urgency condition after it exceeded the original
math-olympiad trial's completion time. No accepted game-CLI state was observed
before cancellation, and the canonical verifier did not run. This is a censored
trial, not a completed score-zero result. It made 204 requests, of which 202
had recorded completions, and recovered from six HTTP 400 responses. All
forwarded requests had the required controls. Its two subagents, client and
relay have stopped. Duration is estimated from the cancellation timestamp;
usage is a lower bound because two streams were unfinished.

The first race attempt started at 08:24:28 local time after urgency cleanup. A live audit
confirmed eight Agent calls in one coordinator response, eight actual starts
spanning 8.9 seconds, eight distinct assigned workdirs with matching original
game input hashes, and no nested agent calls. The coordinator has made zero
game-CLI calls. This attempt then ended at 613.5 seconds without a submission:
Claude Code's print mode automatically killed all eight background agents ten
minutes after the coordinator's final turn. Its native result reports zero
completed agents and eight killed by the system. The raw verifier score was
zero because no answer existed; this is recorded as an invalid setup attempt,
not a completed test of first-finished solving.

The rerun sets `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` only in the private race
client's environment. This disables the print-mode background waiting limit;
the benchmark still stops the whole task after one hour. The model controls,
eight-solver prompt and copied puzzle inputs were unchanged. The corrected run
continued past ten minutes with all eight solvers. One solver returned an
explicitly invalid candidate at about fourteen minutes; the coordinator kept
waiting for the others. No accepted state was observed in the inspected results.

The user then stopped the corrected race at approximately 1,506.3 seconds.
The external verifier did not run, so this is also a censored trial with unknown
final correctness, rather than a completed score-zero result. The benchmark,
Claude client, its remaining agents and temporary relay are stopped. No further
tests are queued. Final input fingerprints are unchanged, including production
and global Claude settings. Both user-canceled trials and the invalid first
race attempt remain available for inspection.

Historical references, each from one earlier trial:

| Condition | Verifier score | Time | Output tokens | Subagents |
| --- | ---: | ---: | ---: | ---: |
| Plain 16K, subagents forbidden | 100 | 989.7 s / 16.5 min | 406,092 | 0 |
| Math-olympiad, original prompt | 100 | 1,468.0 s / 24.5 min | 1,060,954 | 1 |
| No added skill, discretionary delegation | 0, timeout | 3,600.5 s / 60.0 min | ≥4,128,265 | 5 |

The earlier plain no-subagent run remains the fastest observed successful
reference until a new verified result beats it. Single trials, different server
loads, sampling variability and fixed condition order limit causal conclusions.
Token usage for canceled/unfinished streams may lack a final usage record and
will be labeled as a lower bound when applicable.
