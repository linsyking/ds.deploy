# DSH Maxwell level-11 comparison

The benchmark agent `dsh-ds-16k` in the adjacent `ai-sde-benchmark/bench.toml`
runs DSH headless against the existing local SGLang deployment. It uses a fresh
private DSH home for each task; user profiles and production defaults are not
modified. No production restart is required.

## Reproduce

Install the CLI once from the deployment repository:

```sh
npm install --prefix .runs/dsh-benchmark --no-audit --no-fund @deepseek-ai/dsh@0.2.1-rc.5
```

Then, from `../ai-sde-benchmark`:

```sh
python -m bench run -t maxwell-1-11 -a dsh-ds-16k -j 1
```

The command uses `../ds.deploy/dsh_benchmark.py`, the deployment's SGLang Python
environment and `.secrets/api-key`. The benchmark configuration records their
absolute paths for this workspace. The secret is read only by the local relay;
it is not written into the DSH profile, manifest or request metadata.

## Controls

| Setting | Value |
| --- | --- |
| DSH | `@deepseek-ai/dsh@0.2.1-rc.5` |
| pi-ai in the installed dependency lock | `1.5.0` |
| Model | `deepseek-v4.1-flash-1017` |
| Endpoint | Local Anthropic API, `http://10.78.202.30:18000` |
| Thinking cap per response | 16,384 tokens |
| Effort | Max |
| Total output cap per response | 64,000 tokens, including thinking |
| Context capacity | 1,048,576 tokens |
| Nominal automatic compaction threshold | 500,000 tokens |
| Task timeout | 3,600 seconds |

DSH selects `reasoningEffort=max` and budget mode
(`forceAdaptiveThinking=false`). Its inspected pi-ai transport sends
`thinking: {type: enabled, budget_tokens: 16384}` and `max_tokens: 64000`,
while omitting `output_config.effort`. The temporary loopback relay adds
explicit `output_config.effort=max` and enforces the same numeric limits on all
requests owned by this trial. It records received and forwarded controls so the
comparison does not depend on a UI label or an implicit server default.

The DSH threshold is set through `compaction-basic.thresholdRatio` to
`500000 / 1048576`. Its compaction algorithm, token estimator, retained tail,
tool-result pruning, system prompt and native tools remain DSH's. Thus the
nominal threshold matches the Claude trials, but this is a comparison of
harnesses, rather than an isolated change to a single inference setting.
Auxiliary session-title inference is disabled for the one-shot trial.

A real smoke request returned the correct answer and verified HTTP 200,
16,384 thinking tokens, explicit Max effort and 64,000 total output tokens.
Task preconditions and agent configuration loading passed before the full run.

## Trial

The level-11 run started on November 3, 2026 at 21:53:44 local time and
finished at 22:01:35. The canonical verifier rejected the submitted move string:
`1-11 not solved (last state: none)`.

| Harness, 16K thinking / Max effort | Score | Task time | Reported output tokens | Retokenized thinking tokens | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: |
| DSH 0.2.1-rc.5 | **0 / 100** | **471.3 s / 7.9 min** | 255,000 | 249,836 | 0 |
| Earlier Claude Code 2.1.317 reference | **100 / 100** | **989.7 s / 16.5 min** | 406,092 | 398,907 | 0 |

DSH exited normally with code 0, before its one-hour timeout. Its final answer
acknowledged that its saved attempt did not solve the puzzle. The statement in
that answer about being unable to solve "within the time" was the model's own
description; the engine did not time it out. This is an unsuccessful submission,
rather than a successful time reduction.

All 44 model requests completed with HTTP 200 and the required forwarded
controls. Eight thinking blocks were within 16 tokens of the cap; the longest
retokenized thinking block was 16,384 tokens. The largest complete response was
16,531 tokens, and no response stopped because of the 64,000-token output cap.
There were no API/transport errors, no stderr output and no compactions.
Consequently this trial does not test the behavior of DSH's compactor under
pressure or establish an appropriate compaction window.

The visible tool audit found one read call and 42 Bash calls, with no script,
compiler, network or subagent commands in the inspected command patterns.
This is a heuristic audit, not a formal proof of compliance. The saved workdir
contains only the original inputs and `solutions.txt`; all task input bytes
match their originals.

Native benchmark artifacts:
`../ai-sde-benchmark/results/20261103-215344/dsh-ds-16k/maxwell-1-11/`.

Private DSH artifacts:
`.runs/dsh-benchmark/trials/20261103-215344-372102/`.
These contain `manifest.json`, numeric `requests.json`, raw DSH `events.jsonl`,
diagnostics and the private session log. The benchmark's `stdout.log` contains
readable text and tool calls; reasoning stays in the private DSH artifacts.
`summary.json` contains the aggregate usage, verifier result and audit counts;
`cleanup.json` records final process, input and production checks.

The earlier Claude Code 16K/Max reference scored 100 in 989.7 seconds, with
406,092 reported output tokens and no compactions. It ran in a four-arm
concurrent budget batch. The new DSH trial runs alone among our benchmark
clients, so its elapsed time cannot establish an isolated harness speedup.
These are single unseeded trials with different harness prompts and tools.
The result does not establish a general quality ranking between the harnesses.
See `LEVEL11_BUDGET_TEST.md` for the earlier comparison and its limitations.

Final checks confirmed that both owned client processes exited and the
loopback relay stopped listening. The production API returned HTTP 200.
`serving.json` still matches Git HEAD and the saved production configuration;
global Claude settings, task inputs, the added benchmark agent configuration
and the launcher were unchanged during the run.
