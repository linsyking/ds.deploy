#!/usr/bin/env python3
"""Run an isolated DSH headless trial against this deployment.

Install the pinned CLI once with:
  npm install --prefix .runs/dsh-benchmark --no-audit --no-fund @deepseek-ai/dsh@0.2.1-rc.5
Run with .venvs/sglang/bin/python dsh_benchmark.py '<task prompt>'.
The loopback relay sets explicit inference controls and records numeric metadata
for this child only. It never changes production or a user's DSH profile.
"""

import argparse
import datetime
import gzip
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
DSH_VERSION = "0.2.1-rc.5"
THINKING_BUDGET = 16384
MAX_OUTPUT = 64000
CONTEXT_WINDOW = 1048576
COMPACTION_WINDOW = 500000


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def controls(body):
    thinking = body.get("thinking") or {}
    return {
        "model": body.get("model"),
        "thinking": {k: thinking[k] for k in ("type", "budget_tokens") if k in thinking},
        "effort": (body.get("output_config") or {}).get("effort"),
        "max_tokens": body.get("max_tokens"),
        "stream": body.get("stream"),
    }


class Relay(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/v1/messages":
            self.send_error(404)
            return
        server = self.server
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        body = json.loads(raw)
        record = {
            "started_at": datetime.datetime.now().isoformat(),
            "received": controls(body),
            "message_count": len(body.get("messages", [])),
        }
        # pi-ai's budget mode omits effort. Set it explicitly for this trial,
        # and fix the total output cap independently of its answer allowance.
        body["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
        body.setdefault("output_config", {})["effort"] = "max"
        body["max_tokens"] = MAX_OUTPUT
        record["forwarded"] = controls(body)
        with server.lock:
            server.records.append(record)
            write_json(server.artifacts / "requests.json", server.records)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in {
            "host", "content-length", "connection", "transfer-encoding",
            "content-encoding", "accept-encoding", "authorization", "x-api-key",
        }}
        headers.update({"Authorization": "Bearer " + server.api_key,
                        "Accept-Encoding": "identity"})
        request = urllib.request.Request(server.upstream + self.path,
            data=json.dumps(body).encode(), headers=headers, method="POST")
        started = time.monotonic()
        thinking_parts = []
        sent_headers = False
        try:
            try:
                response = server.opener.open(request, timeout=900)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                record["http_status"] = response.status
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in {"connection", "transfer-encoding", "content-length", "keep-alive"}:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                sent_headers = True
                for line in response:
                    if line.startswith(b"data:"):
                        try:
                            event = json.loads(line[5:])
                            if event.get("type") == "message_start":
                                message = event.get("message", {})
                                record["response_model"] = message.get("model")
                                record["usage"] = message.get("usage", {})
                            elif event.get("type") == "message_delta":
                                record["stop_reason"] = event.get("delta", {}).get("stop_reason")
                                record.setdefault("usage", {}).update(event.get("usage", {}))
                            elif event.get("type") == "content_block_delta":
                                part = event.get("delta", {}).get("thinking")
                                if part:
                                    thinking_parts.append(part)
                            elif event.get("type") == "error":
                                record["stream_error_type"] = event.get("error", {}).get("type")
                        except (ValueError, AttributeError):
                            pass
                    self.wfile.write(line)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            record["client_disconnected"] = True
        except Exception as error:
            record["transport_error_type"] = type(error).__name__
            if not sent_headers:
                self.send_error(502, "Isolated DSH relay transport failed")
        finally:
            record["seconds"] = round(time.monotonic() - started, 3)
            record["thinking_text_tokens"] = len(server.tokenizer.encode(
                "".join(thinking_parts), add_special_tokens=False).ids)
            with server.lock:
                write_json(server.artifacts / "requests.json", server.records)


def make_patch(model, base_url):
    return [
        {"id": "agent-default-model", "config": {
            "provider": "local-sglang", "model": model, "reasoningEffort": "max"}},
        {"id": "llm-pi-ai", "config": {"providers": {"local-sglang": {
            "apiKeyEnv": "DSH_BENCH_API_KEY", "api": "anthropic-messages",
            "baseURL": base_url, "reasoning": "max",
            "thinkingBudgets": {"high": THINKING_BUDGET},
            "compat": {"forceAdaptiveThinking": False, "allowEmptySignature": True},
            "models": [{"id": model, "contextWindow": CONTEXT_WINDOW,
                        "maxTokens": MAX_OUTPUT, "input": ["text"],
                        "reasoningEfforts": {"off": None, "high": "high", "max": "max"}}],
        }}}},
        {"id": "compaction-basic", "config": {
            "thresholdRatio": COMPACTION_WINDOW / CONTEXT_WINDOW,
            "maxTokens": MAX_OUTPUT}},
        # Auxiliary title calls have a different output budget and add no value
        # to a one-shot benchmark. Keep them outside the measured trial.
        {"id": "session-title-llm", "disabled": True},
    ]


def render(event):
    kind = event.get("type")
    if kind in {"text", "final"}:
        return event.get("text", "")
    if kind == "tool_call":
        value = json.dumps(event.get("input", {}), ensure_ascii=False)
        return f'[{event.get("tool", "tool")}] {value[:600]}'
    if kind == "tool_result" and event.get("status") == "error":
        return "[tool error] " + event.get("result", "")[:600]
    if kind == "error":
        return "[error] " + event.get("message", "")
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    install = ROOT / ".runs/dsh-benchmark"
    package = install / "node_modules/@deepseek-ai/dsh"
    if not (package / "package.json").exists():
        parser.error("Install the pinned DSH package using the command in this module's docstring")
    if json.loads((package / "package.json").read_text())["version"] != DSH_VERSION:
        parser.error("The isolated DSH installation does not match the pinned version")
    cfg = json.loads((ROOT / "serving.json").read_text())
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    artifacts = (args.artifacts or install / "trials" / stamp).resolve()
    artifacts.mkdir(mode=0o700, parents=True, exist_ok=False)
    private = artifacts / "dsh-home"
    private.mkdir(mode=0o700)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.daemon_threads = True
    server.artifacts = artifacts
    server.upstream = f'http://{cfg["host"]}:{cfg["port"]}'
    server.api_key = (ROOT / ".secrets/api-key").read_text().strip()
    server.tokenizer = Tokenizer.from_file(str(Path(cfg["model_path"]) / "tokenizer.json"))
    server.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server.lock = threading.Lock()
    server.records = []
    patch = artifacts / "profile.patch.yml"
    write_json(patch, make_patch(cfg["served_model_name"], f"http://127.0.0.1:{server.server_port}"))
    pi_ai_version = json.loads((install / "node_modules/@earendil-works/pi-ai/package.json").read_text())["version"]
    manifest = {"dsh_version": DSH_VERSION, "pi_ai_version": pi_ai_version,
        "upstream": server.upstream, "model": cfg["served_model_name"],
        "thinking_budget": THINKING_BUDGET, "effort": "max", "max_output": MAX_OUTPUT,
        "context_window": CONTEXT_WINDOW, "compaction_window": COMPACTION_WINDOW,
        "workdir": str(Path.cwd()), "started_at": datetime.datetime.now().isoformat(),
        "serving_sha256": hashlib.sha256((ROOT / "serving.json").read_bytes()).hexdigest(),
        "notes": "DSH budget mode; relay explicitly sets effort=max, budget=16384, max_tokens=64000. Native DSH tools and compaction algorithm."}
    write_json(artifacts / "manifest.json", manifest)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "DSH_"))}
    env.update(DSH_HOME=str(private), DSH_BENCH_API_KEY="isolated-relay",
               DSH_PERMISSION_MODE="danger-full-access", DSH_TELEMETRY_DISABLED="1")
    command = [shutil.which("node") or "node", str(package / "lib/bin.js"),
               "--profile", "headless", "--patch", str(patch), "--json", args.prompt]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    child = None
    started = time.monotonic()

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print("DSH artifacts: " + str(artifacts), flush=True)
    try:
        with (artifacts / "events.jsonl").open("w") as events, (artifacts / "stderr.log").open("w") as errors:
            child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=errors,
                                     text=True, bufsize=1)
            write_json(artifacts / "process.json", {"wrapper_pid": os.getpid(), "dsh_pid": child.pid,
                "relay_port": server.server_port})
            for line in child.stdout:
                events.write(line)
                events.flush()
                try:
                    event = json.loads(line)
                    readable = render(event)
                except (ValueError, AttributeError):
                    readable = line.rstrip()
                if readable:
                    print(readable, flush=True)
            code = child.wait()
        return code
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
        write_json(artifacts / "completion.json", {
            "exit_code": child.returncode if child is not None else None,
            "duration_s": round(time.monotonic() - started, 3),
            "serving_unchanged": manifest["serving_sha256"] == hashlib.sha256((ROOT / "serving.json").read_bytes()).hexdigest()})


if __name__ == "__main__":
    sys.exit(main())
