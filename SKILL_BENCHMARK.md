# Claude Code skill comparison on Maxwell level 11

This experiment keeps the local DeepSeek / Claude Code 16K setup and changes the
selected reasoning skill. Trials run **sequentially**, with at most **eight total
subagent spawns per trial**; work inside a trial may run in parallel.

## Conditions

All conditions use Claude Code 2.1.317, `deepseek-v4.1-flash-1017[1m]`, explicit
16,384 thinking tokens and Max effort on every owned model request, 64,000 total
output tokens per response, a nominal 500K compaction window, and the canonical
Maxwell level-11 game and verifier. Each task has the normal one-hour timeout.

The order is the four requested skills followed by a fresh no-skill control:

1. [math-olympiad](https://github.com/anthropics/claude-plugins-official/tree/fa4eb54397ee2faccb6940512a19c017cb1337f3/plugins/math-olympiad/skills/math-olympiad), revision `fa4eb54397ee2faccb6940512a19c017cb1337f3`.
2. [math-proof](https://github.com/flonat/flonat-research/tree/08dd9bdf9367f7321c27b7db8c08a125598a613f/skills/math-proof), revision `08dd9bdf9367f7321c27b7db8c08a125598a613f`.
3. [atom-of-thought](https://github.com/benjaminjackson/reasoning-skills/tree/c72f28b0fe0c7b37883fd7921a8d4251fc02eb23/reasoning/skills/atom-of-thought), revision `c72f28b0fe0c7b37883fd7921a8d4251fc02eb23`. The user selected this skill from the four-skill repository.
4. [structured-brainstorming](https://github.com/oborchers/fractional-cto/tree/182a2e1e6033cbcd2dd086a1d6bcee956dc901f6/structured-brainstorming/skills/structured-brainstorming), revision `182a2e1e6033cbcd2dd086a1d6bcee956dc901f6`.
5. No external skill, with the same eight-agent allowance and common instructions.

The control is needed because the earlier successful 16K reference forbade
subagents. That reference also ran under four-arm concurrent load, so its
989.7-second time is historical context rather than a matched control here.

## Isolation and verification

`claude_skill_benchmark.py` creates a fresh private Claude profile for each task.
It installs only that condition's external skill, including its bundled
references, and activates it through its native slash command. Claude's built-in
skills remain available in all conditions. The brainstorming condition also
includes its upstream `brainstorm-explorer` agent definition. No other skills
from the reasoning-skills repository are installed.

The source skill files are unmodified. Common instructions make the puzzle
objective and rules take precedence over conflicting skill suggestions: no
internet, scripts, solvers or game-source reverse engineering. The requested
submission remains `solutions.txt`; a proof, PDF or brainstorming report does
not substitute for a game solution. User authorization covers dispatch in this
noninteractive test. Model-tier requests are mapped to the same local DeepSeek
model, keeping the model consistent across parent and children.

In particular, math-olympiad's suggested 8–12 initial solvers plus later
verifiers exceed the user-requested total cap. The model must allocate the eight
slots across phases. These are bounded adaptations to the game and the stated
resource limit, rather than unrestricted executions of each skill's defaults.

A private task copy changes only `allow_subagents` to true; original task inputs
and its verifier are preserved. All task workdirs and client profiles are fresh,
auto-memory is disabled, and agents are told not to consult previous trials.
The production service and global Claude settings are not modified.

`agent_limit_hook.py` uses a file lock and shared per-trial ledger to reserve at
most eight new Agent/Task calls across parent and nested agents. Resuming an
existing agent does not consume a new slot. The ledger also records actual
subagent starts/stops. A concurrent cap check allowed exactly eight of 24
attempts, rejecting the other 16. A real Claude Code smoke test successfully
started two children; a separate real deny test blocked spawning and started
zero children. Native skill expansion and forwarded inference controls were
also verified before starting the benchmark.

The temporary loopback relay records numerical limits, usage, timing and stop
reasons for this client's requests. It supplies explicit enabled thinking,
16,384 budget tokens, Max effort and 64,000 total output tokens to both the
parent and children. Output usage includes all these streams and any compaction
requests. Retokenized thinking is an estimate from streamed text, not a distinct
server reasoning-usage counter.

## Run

The sequential queue started November 3, 2026 at 22:18:35 local time:

`.runs/skill-tests/20261103-221835-sequential/`.

The queue finished November 4 at 01:25:46 local time. `state.json` records all
five conditions complete; `comparison.json` contains their results.
Each condition has a private benchmark config, standard
benchmark result/workdir, relay metadata, Claude transcript, cap ledger and
completion summary. `.runs/skill-tests/sources.json` records exact source
revisions and hashes of every installed skill file.

```sh
.venvs/sglang/bin/python -u .runs/skill-tests/run_sequential.py
```

## Results

| Condition | Score | Time | Output tokens | Thinking tokens | Subagents | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math-olympiad | 100 | 1,468.0 s / 24.5 min | 1,060,954 | 1,038,007 | 1 | 1 |
| math-proof | 100† | 1,999.8 s / 33.3 min | 1,899,787 | 1,820,041 | 2 | 3 |
| atom-of-thought | 100‡ | 1,979.4 s / 33.0 min | 1,845,231 | 1,795,428 | 2 | 2 |
| structured-brainstorming | 100§ | 2,176.0 s / 36.3 min | 2,873,724 | 2,805,666 | 4 | 3 |
| No-skill control | 0¶ | 3,600.5 s / 60.0 min, timeout | ≥4,128,265 | ≥4,011,456 | 5 | 6 |

Math-olympiad was the fastest accepted submission, used the fewest output
tokens among accepted submissions, and was the only accepted run without
tool-audit flags. Treat it as the strongest candidate from this screen, rather
than a proven general improvement. The control did not solve the level before
its deadline, so there is no successful matched-control solving time from which
to calculate a speedup. The older no-agent 16K reference solved it in 989.7
seconds (16.5 minutes), faster than any run here, under different conditions.
This screen therefore does not establish a new best overall time.

Math-olympiad exited normally and passed the canonical verifier. All 112 model
requests had HTTP 200 and verified controls. Its visible tool audit found no
script, compiler, network or binary-inspection commands in the inspected
patterns, and all input files in the saved workdir match their originals.
This is a heuristic audit rather than a formal proof of compliance.
It used one subagent, including resumptions, rather than the skill's full
suggested solver/verifier fan-out. Its automatic compaction took 18.198 seconds,
reducing approximately 478,630 to 26,542 retained tokens.

† Math-proof's move string passed the canonical game verifier, but its tool log
contains a Python snippet used to parse its own subagents' output logs. That
violates the strict no-scripts instruction; this run should not be ranked as a
fully compliant success. The snippet performed log bookkeeping, rather than
game-state search. No solver program or source/binary inspection was found in
the inspected patterns, and its saved game inputs match the originals.
It had 244 HTTP 200 responses and three recovered HTTP 400 responses near the
end (0.074 seconds aggregate recorded relay time for the 400s). The final client
exit was normal and every forwarded request had the required controls. Its
three compactions took 18.751, 14.595 and 13.436 seconds. Its client and relay
have stopped.

‡ Atom-of-thought passed the canonical verifier, with 199 HTTP 200 responses,
verified controls, two subagents and two compactions (20.423 and 11.859 seconds).
Its audit flags one command invoking `curl` against `http://127.0.0.1:1/` before
running the game CLI. This was a loopback probe; there is no evidence of an
external web fetch or a solver program. It is nevertheless an extra tool under
the task's strict tool restrictions, so this result also carries a compliance
qualification. Original game inputs are unchanged, and its client and relay
have stopped.

§ Structured-brainstorming passed the canonical verifier, with 310 HTTP 200
responses, verified controls, four subagents and three compactions (18.466,
23.103 and 48.392 seconds). Its audit flags an invocation of `python3 -I -`
with an empty here-document while inspecting its own game-output log using
`grep`. No Python code was supplied and no solver computation was found;
this extra-tool invocation is still recorded under the strict restrictions.
Its game inputs are unchanged, and its client and relay have stopped.

¶ The no-external-skill control reached the normal one-hour timeout. Its saved
candidate failed canonical verification: `1-11 not solved (last state: none)`.
It spawned five subagents and compacted six times (100.808 seconds aggregate
compaction duration). Its tool audit also flags Python snippets for reading a
subagent log and printing the first three whitespace-delimited tokens of the
raw level file; both carry the strict no-scripts/tool-rule qualification.
No solver algorithm or game-source/binary inspection was found in the inspected
patterns. All original game inputs remain unchanged.

The control made 518 model requests, of which 514 completed with HTTP 200;
four were still in flight when the client was stopped at its time limit.
Its output and retokenized-thinking figures are lower bounds because those
unfinished streams have no final usage record. Its client exit code 143 is
expected termination at the benchmark deadline, rather than a setup failure.
All forwarded request controls were verified, including the unfinished calls.

The first observed winning game-CLI states occurred at approximately 1,240.0
seconds for math-olympiad, 1,984.3 seconds for math-proof, and 1,526.5 seconds
for atom-of-thought, and 1,800.1 seconds for structured-brainstorming. Their final
exits were another 228.0, 15.5, 452.9 and 375.9 seconds later, respectively.
Total runtime remains the primary metric; these timestamps separately expose
post-solution overhead. No accepted state was observed in the control's inspected
game-CLI results.

All five clients, wrappers and temporary relays have stopped. The coordinator
finished normally. Final input fingerprints match the pre-run fingerprints:
serving configuration, global Claude settings, benchmark configuration, original
level files, skill sources and benchmark runtime scripts are unchanged. The
production `/health` endpoint returned HTTP 200 after the queue finished.

One unseeded trial per condition provides a provisional comparison. Agent usage
may differ by skill, and the fixed order, cache state and unrelated service
traffic remain uncontrolled factors. Shorter time with a failed solution is not
a successful solving-time improvement.
