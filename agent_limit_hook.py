#!/usr/bin/env python3
"""Private Claude Code hook: bound new Agent/Task calls across a whole trial."""
import fcntl
import json
from pathlib import Path
import sys
import time


def handle(directory, event, limit=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "agent-limit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "agent-limit.json"
        state = json.loads(path.read_text()) if path.exists() else {
            "limit": limit, "reservations": {}, "events": [], "skill_calls": []}
        hook = event.get("hook_event_name")
        tool = event.get("tool_name")
        params = event.get("tool_input") or {}
        output = {}
        if hook == "PreToolUse" and tool in {"Agent", "Task"}:
            key = event.get("tool_use_id") or f"missing-{time.time_ns()}"
            resume = bool(params.get("resume"))
            allowed = resume or key in state["reservations"] or len(state["reservations"]) < state["limit"]
            if allowed and not resume:
                state["reservations"][key] = {"parent_agent_id": event.get("agent_id"),
                    "requested_type": params.get("subagent_type"), "time": time.time()}
            state["events"].append({"hook": hook, "tool_use_id": key,
                "allowed": allowed, "resume": resume, "time": time.time()})
            if not allowed:
                output = {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                    "permissionDecision": "deny", "permissionDecisionReason":
                    f"This trial permits at most {state['limit']} total subagent spawns, including solvers, verifiers and revisers. Reuse existing agents or continue in the main agent."}}
        elif hook == "PreToolUse" and tool == "Skill":
            state["skill_calls"].append({"skill": params.get("skill"), "time": time.time()})
        elif hook in {"SubagentStart", "SubagentStop"}:
            state["events"].append({"hook": hook, "agent_id": event.get("agent_id"),
                "agent_type": event.get("agent_type"), "time": time.time()})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)
        return output


if __name__ == "__main__":
    try:
        result = handle(sys.argv[1], json.load(sys.stdin), int(sys.argv[2]) if len(sys.argv) > 2 else 8)
        print(json.dumps(result))
    except Exception:
        # A broken cap hook must block spawning rather than silently lose the cap.
        print("Cannot verify this trial's subagent limit.", file=sys.stderr)
        sys.exit(2)
