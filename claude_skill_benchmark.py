#!/usr/bin/env python3
"""An isolated local Claude Code 16K/Max trial with one selected skill."""
import argparse
import ast
import datetime
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from tokenizers import Tokenizer

from dsh_benchmark import ROOT, Relay, write_json

SKILLS = ROOT / ".runs/skill-tests/skills"
ARMS = ["math-olympiad", "math-proof", "atom-of-thought", "structured-brainstorming", "baseline"]
RULES = """The user has authorized this noninteractive benchmark and subagent dispatch.
Solve the level described in readme.txt and save the required move string in solutions.txt.
You may spawn AT MOST EIGHT TOTAL subagents across this entire run, including all
solvers, verifiers, revisers and nested agents. Budget those eight across phases;
reuse agents when useful. All agents use the same local DeepSeek model, Max effort,
16,384 thinking tokens and 64,000 total output tokens per response.
The benchmark's game rules take precedence over incompatible skill suggestions:
no internet, no source reverse engineering, no programs, scripts or solvers.
Use ./maxwell as the only way to play and test. Reading task instructions and
skill references, delegating reasoning, and writing the answer are permitted.
Apply the selected skill's reasoning workflow to this game, then complete the game
task; a proof document, brainstorming recommendation or PDF is not the submission.
User approval to dispatch is already granted; proceed without interactive questions.
Do not use results, solutions, transcripts or memory from any previous trial.
"""


def race_prompt(workdirs):
    workers = "\n".join(f"- Solver {i}: {path}" for i, path in enumerate(workdirs, 1))
    return f"""\n\nPARALLEL INDEPENDENT SOLVER RACE, N={len(workdirs)}.
You are only the coordinator. Immediately launch {len(workdirs)} general-purpose
subagents concurrently, preferably all Agent calls in one response. Each must
independently solve the ENTIRE level, rather than a different subproblem. No skill
is selected. All eight spawn slots belong to these solvers; no nested agents.
Each solver has its own fresh working copy:
{workers}
Give every solver these instructions, including its absolute working directory:
Read readme.txt in YOUR directory, solve the entire level independently using
only ./maxwell, verify State: accept with the game CLI, write solutions.txt in
YOUR directory, then immediately return its absolute path and complete move
string. Keep trying until solved or stopped. Do not spawn agents, use any skills,
inspect another solver's files or messages, use earlier trials, write programs
or scripts, search the internet, or inspect raw level/game internals. Do not write
the coordinator's solutions.txt. Every Bash game call must cd to YOUR directory.

The coordinator must NOT solve, plan moves, explore the game, improve candidates,
or assign different reasoning strategies. After dispatch, just wait for completion
notifications. If necessary, use sleep and ListAgents sparingly to wait; keep
coordinator responses brief. Do not parse JSON logs with scripts. Do not block
waiting for one particular solver while other solvers could finish first.
On the FIRST solver completion reporting a complete, CLI-verified solution,
immediately copy that solver's solutions.txt to the coordinator's solutions.txt,
stop the other solvers with TaskStop, and finish with a brief response. Do not
wait for the rest, compare answers, run another solver, or rework the candidate.
The benchmark's external verifier will judge the submitted answer. If a solver
returns without a complete candidate, continue waiting for the others.
"""


def prepare_prompt(prompt, skill, suffix="", workdirs=()):
    prompt = prompt.replace(" Do not spawn any subagents.", "")
    if suffix:
        prompt += "\n\n" + suffix
    if workdirs:
        prompt += race_prompt(workdirs)
    if skill != "baseline":
        prompt = f"/{skill} {prompt}\n\nApply this skill to solve the puzzle."
    return prompt


def transcript_stats(private):
    tools = {}
    compact = []
    for path in (private / "projects").rglob("*.jsonl"):
        for line in path.open(errors="replace"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("subtype") == "compact_boundary":
                compact.append({k: row[k] for k in ["timestamp", "compactMetadata"] if k in row})
            if row.get("type") == "assistant":
                for block in row.get("message", {}).get("content", []):
                    if block.get("type") == "tool_use":
                        tools[block["id"]] = block.get("name")
    return {"compaction_count": len(compact), "compaction_events": compact,
            "tool_counts": {name: list(tools.values()).count(name) for name in sorted(set(tools.values()))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=ARMS, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--max-agents", type=int, choices=range(9), default=8)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--race-solvers", type=int, choices=range(9), default=0)
    parser.add_argument("prompt")
    args = parser.parse_args()
    if args.race_solvers and (args.skill != "baseline" or not 2 <= args.race_solvers <= args.max_agents):
        parser.error("A solver race requires baseline skill and 2..max-agents solvers")
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(mode=0o700, parents=True, exist_ok=False)
    private = artifacts / "claude-config"
    private.mkdir(mode=0o700)
    if args.skill != "baseline":
        source = SKILLS / args.skill
        if not (source / "SKILL.md").exists():
            parser.error("The selected skill has not been installed in the private source cache")
        shutil.copytree(source, private / "skills" / args.skill)
    if args.skill == "structured-brainstorming":
        (private / "agents").mkdir()
        shutil.copy2(ROOT / ".runs/skill-tests/source-brainstorming/structured-brainstorming/agents/brainstorm-explorer.md",
                     private / "agents/brainstorm-explorer.md")
    cfg = json.loads((ROOT / "serving.json").read_text())
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.daemon_threads = True
    server.artifacts = artifacts
    server.upstream = f'http://{cfg["host"]}:{cfg["port"]}'
    server.api_key = (ROOT / ".secrets/api-key").read_text().strip()
    server.tokenizer = Tokenizer.from_file(str(Path(cfg["model_path"]) / "tokenizer.json"))
    server.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server.lock = threading.Lock()
    server.records = []
    model = cfg["served_model_name"] + "[1m]"
    hook = " ".join(map(shlex.quote, [sys.executable, str(ROOT / "agent_limit_hook.py"), str(artifacts), str(args.max_agents)]))
    settings = {
        "autoMemoryEnabled": False,
        "env": {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            "ANTHROPIC_AUTH_TOKEN": "isolated-benchmark-relay",
            "ANTHROPIC_API_KEY": "isolated-benchmark-relay",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
            "ENABLE_TOOL_DISCOVERY": "1",
        },
        "hooks": {
            "PreToolUse": [{"matcher": "^(Agent|Task|Skill)$", "hooks": [{"type": "command", "command": hook}]}],
            "SubagentStart": [{"hooks": [{"type": "command", "command": hook}]}],
            "SubagentStop": [{"hooks": [{"type": "command", "command": hook}]}],
        },
    }
    write_json(private / "settings.json", settings)
    original = Path.home() / ".claude.json"
    if original.exists():
        shutil.copy2(original, private / ".claude.json")
        (private / ".claude.json").chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_"))}
    env.update(settings["env"])
    env.update(CLAUDE_CONFIG_DIR=str(private), CLAUDE_CODE_DISABLE_AUTO_MEMORY="1",
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    if args.race_solvers:
        # A coordinator that only waits makes no new turns. Print mode otherwise
        # kills its still-working children after ten idle minutes.
        env["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] = "0"
    # A private task copy enables subagents. The replacement also makes a direct
    # invocation with the ordinary task prompt honor the user's explicit override.
    workdirs = []
    for i in range(1, args.race_solvers + 1):
        directory = Path.cwd() / "race-solvers" / f"solver-{i:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(Path.cwd() / "readme.txt", directory / "readme.txt")
        shutil.copy2(Path.cwd() / "maxwell", directory / "maxwell")
        shutil.copytree(Path.cwd() / "levels", directory / "levels")
        workdirs.append(directory)
    prompt = prepare_prompt(args.prompt, args.skill, args.prompt_suffix, workdirs)
    disallowed = "WebSearch,WebFetch,AskUserQuestion"
    if args.race_solvers:
        disallowed += ",Skill,Workflow"
    command = [shutil.which("claude") or "claude", "--model", model, "-p", prompt,
        "--effort", "max", "--autocompact", "500k", "--dangerously-skip-permissions",
        "--output-format", "stream-json", "--verbose", "--setting-sources", "user",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--disallowedTools", disallowed,
        "--append-system-prompt", RULES]
    inputs = [ROOT / "serving.json", Path.home() / ".claude/settings.json"]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest = {"skill": args.skill, "thinking_budget": 16384, "effort": "max",
        "max_output": 64000, "compact": "500k", "max_total_subagents": args.max_agents,
        "upstream": server.upstream, "model": model, "workdir": str(Path.cwd()),
        "started_at": datetime.datetime.now().isoformat(), "input_sha256": before,
        "prompt_suffix": args.prompt_suffix, "race_solvers": args.race_solvers,
        "background_wait_ceiling_ms": 0 if args.race_solvers else None,
        "solver_workdirs": list(map(str, workdirs)),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "skill_sha256": {str(p.relative_to(private)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (private / "skills").rglob("*") if p.is_file()},
        "claude_version": subprocess.check_output([command[0], "--version"], text=True).strip()}
    write_json(artifacts / "manifest.json", manifest)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    child = None
    start = time.monotonic()

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print("Claude skill artifacts: " + str(artifacts), flush=True)
    try:
        with (artifacts / "events.jsonl").open("w") as events, (artifacts / "stderr.log").open("w") as errors:
            child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=errors,
                                     text=True, bufsize=1)
            write_json(artifacts / "process.json", {"wrapper_pid": os.getpid(), "claude_pid": child.pid,
                "relay_port": server.server_port})
            for line in child.stdout:
                events.write(line)
                events.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return child.wait()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        server.shutdown()
        server.server_close()
        with server.lock:
            write_json(artifacts / "requests.json", server.records)
        summary = {"exit_code": child.returncode if child is not None else None,
            "duration_s": round(time.monotonic() - start, 3),
            "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in server.records),
            "thinking_text_tokens": sum(r.get("thinking_text_tokens", 0) for r in server.records),
            "completed_responses": sum("seconds" in r for r in server.records),
            "near_thinking_cap_responses": sum(r.get("thinking_text_tokens", 0) >= 16368 for r in server.records),
            "controls_verified": bool(server.records) and all(r["forwarded"]["thinking"] == {"type": "enabled", "budget_tokens": 16384}
                and r["forwarded"]["effort"] == "max" and r["forwarded"]["max_tokens"] == 64000
                and r["forwarded"]["model"] == cfg["served_model_name"] for r in server.records),
            "http_errors": [r.get("http_status") for r in server.records if r.get("http_status", 200) != 200],
            "changed_inputs": [str(p) for p in inputs if hashlib.sha256(p.read_bytes()).hexdigest() != before[str(p)]]}
        summary.update(transcript_stats(private))
        ledger = artifacts / "agent-limit.json"
        if ledger.exists():
            state = json.loads(ledger.read_text())
            summary.update(subagent_reservations=len(state["reservations"]),
                subagents_started=len({e["agent_id"] for e in state["events"] if e["hook"] == "SubagentStart"}),
                denied_agent_calls=sum(e.get("allowed") is False for e in state["events"]))
        write_json(artifacts / "completion.json", summary)


if __name__ == "__main__":
    sys.exit(main())
