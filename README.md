# DeepSeek V4.1 Flash on c04

Single-node deployment of deepseek-ai/DeepSeek-V4.1-Flash-1017 on c04
(10.78.202.32), using four GB200 GPUs and SGLang 0.5.23.
Checkpoint revision: 1bae0e1088993b9b82de08f59de5be5a34e42170.
The downloaded checkpoint is hash-verified; native vision and DSpark draft
weights are included. The model path and tunable values are in serving.json.

## Operations

Run these commands in this repository on the control host or c04:

    ./ds41ctl start
    ./ds41ctl status
    ./ds41ctl logs --follow
    ./ds41ctl restart
    ./ds41ctl stop
    ./ds41ctl load
    ./sglang_load.py 2
    ./ds41ctl check
    ./ds41ctl check-reasoning

Start/restart sync the small runtime files to c04. The engine runs in a named
tmux session and survives SSH disconnection. These commands only signal this
controller's tracked process group. They refuse to start if another GPU job is
present or a GPU reports a recovery action. Startup includes weight loading,
CUDA graph capture and native vision warmup; allow several minutes.

Edit serving.json and restart to change engine settings. The service uses the
installed .venvs/sglang environment, its CUDA 13 toolchain, and node-local
scratch caches under .runs/kernel-cache/sglang. Model files are at
/scratch_local/user_data/yiming/models/DeepSeek-V4.1-Flash-1017.
GPU reset/reboot and automatic crash-restart are deliberately not performed.
A machine reboot requires starting the service and tunnel again.

## API and vision

Internal OpenAI-compatible base URL: http://10.78.202.32:18000/v1
Anthropic-compatible base URL: http://10.78.202.32:18000
Model name: deepseek-v4.1-flash-1017

Use the separate API credential in .secrets/api-key. For Claude Code, set
ANTHROPIC_BASE_URL to the Anthropic base URL, ANTHROPIC_AUTH_TOKEN to that key,
and select the model alias above. Use bearer authentication; the installed
native middleware checks Authorization, not x-api-key. Native vision accepts
OpenAI image_url content or Anthropic base64 image content. No text-only
middleware is installed. Reasoning and structured tool calls use the native
DeepSeek codec and auto parsers. Reasoning defaults to ON with max effort (numeric effort 100). Explicit
per-request settings still take precedence.

### Reasoning defaults

serving.json contains default_thinking=true and default_reasoning_effort="max".
The launcher passes these through SGLang's native --default-chat-template-kwargs.
This is necessary because Anthropic output_config.effort="max" alone previously
left thinking at the engine's off default. The engine package itself is unmodified.

Requests that omit the thinking toggle now reason at max effort. Anthropic
thinking.type="disabled" and OpenAI reasoning_effort="none" still turn thinking
off. Explicit low/high effort remains supported. No Claude Code settings are changed.
These defaults take effect after a service restart. ./ds41ctl check-reasoning
checks actual thinking output, streaming, off overrides, and equality between the
server-default prompt tokens and an explicitly configured prompt.

For Claude Code / Anthropic, off means thinking.type="disabled". Effort levels
map to low=50, medium=75, high=75, max=100. OpenAI uses reasoning_effort="none"
for off and the same native values for low/medium/high/max. The extra xhigh alias
maps to 75 on OpenAI Chat Completions and 100 on Anthropic Messages; use explicit
low/high/max for consistent behavior across the two protocols.

Both APIs reach the same model. OpenAI Chat Completions returns reasoning_content;
Anthropic Messages returns thinking content blocks. Anthropic adaptive thinking is
implemented as enabled thinking, not Claude's adaptive policy. Anthropic
thinking.budget_tokens is accepted for SDK compatibility but is not an independent
hard cap; max_tokens limits total generated reasoning plus answer tokens.

Core text, reasoning, tool calls/round trips, images and streaming have been checked.
This does not certify every OpenAI/Anthropic endpoint or provider-specific feature;
Responses and cloud-service APIs are outside this compatibility test scope.
One random streaming-image content assertion failed without a retained response;
retesting a known fixture through OpenAI streaming/non-streaming and Anthropic
streaming succeeded. Protocol compatibility does not guarantee recognition accuracy.
Details are in audit/reasoning-check-20261031-154801/.

DSH with an Anthropic-compatible provider needs an explicit adaptive-thinking
compatibility setting for effort levels. Without it, the current DSH pi-ai
transport sends thinking.budget_tokens (medium=8192, max=16384) and omits
output_config.effort. SGLang does not enforce those budgets, so both use this
deployment's default max effort. In the existing model entry in the active
$DSH_HOME/profiles/<profile>/cordis.patch.yml, merge these fields while preserving
the other model and provider settings:

```yaml
- id: deepseek-v4.1-flash-1017
  compat:
    forceAdaptiveThinking: true
  reasoningEfforts:
    off:
    low: low
    medium: medium
    high: high
    max: max
```

The current transport then sends thinking.type=adaptive with
output_config.effort for enabled levels, and thinking.type=disabled for off.
The native effort mapping remains low=50, medium/high=75, max=100. If an older
DSH version rejects forceAdaptiveThinking, use a version supporting this setting.
The configuration takes effect on the next request; no SGLang restart is needed.

A local reproduction using the pi-ai 1.3.0 dependency pinned by current DSH,
including its upstream DSH patch, verified both request fields and streamed
output against c04. With adaptive mode, off returned zero reasoning characters,
medium returned 1050, and max returned 2408 on one synthetic prompt. These are
diagnostics, not latency or quality guarantees. The user's inaccessible DSH
installation and its reported identical 39-second times have not been inspected;
the off failure did not reproduce in the current transport. Evidence is in
audit/dsh-reasoning-20261031-155906/.

The published latency comparison used non-thinking synthetic token-ID prompts.
Max thinking can generate additional reasoning tokens and delay the visible final
answer; the previous benchmark is not a max-thinking latency guarantee.

The context limit is 1,048,576 tokens, including prompt and generated output.
The intended workload is one or two users below 500K tokens, with 1M tested
only for one request. max_running_requests is a scheduling cap, not a promise
that every request length fits at that concurrency. Actual cache reuse depends
on request prefixes and the hybrid cache's eviction state.

./ds41ctl check uses an image with a random code, tests both vision APIs, and
checks an Anthropic tool-call/result round trip, including streaming. Evidence is saved in audit/.

## Remote tunnel

    ./ds41ctl tunnel-start
    ./ds41ctl tunnel-status
    ./ds41ctl tunnel-stop

On your server (ssh.linsyking.com), use:
OpenAI base URL: http://127.0.0.1:8312/v1
Anthropic base URL: http://127.0.0.1:8312

Server-local access is the requested scope. External access to port 8312 is
not required, and no server firewall or public exposure change was made.

FRPC connects to ssh.linsyking.com:17000 through the loopback HTTP proxy
http://127.0.0.1:18080. GOST uses Cloudflare DNS over HTTPS:

    nohup gost -L 'http://127.0.0.1:18080?resolver=https://1.1.1.1/dns-query' >/dev/null 2>&1 &

The TCP mapping is remote port 8312 to c04's actual bound API
10.78.202.32:18000. The sample's local port 8000 is not used. The FRP token is
in .secrets/frpc.toml, separate from the API key. Private files have mode 0600
inside a mode-0700 directory and are ignored by Git. Credentials are not
included in normal runtime sync; they are provisioned separately over SSH.
FRPC runs under nohup in its own process group, reconnects after connection
loss, and logs to .runs/frpc.log. tunnel-stop stops only owned FRPC; GOST is
left available. tunnel-start refuses to expose an API without authentication.
FRP transport TLS protects c04-to-FRPS traffic. The server-local API uses HTTP
and requires the same API key. FRPC login and proxy registration are verified;
a loopback request on the remote server requires running a client there.

GOST 3.3.0 was already installed. FRPC 0.71.1 linux/arm64 was downloaded from
the official fatedier/frp release and verified against its SHA-256 checksum.

## Evidence and maintenance

DEPLOYMENT_STATUS.md records final checks and measured settings. The selected
16K prefill chunk reduced 480K-token cold TTFT from 12.37s to 4.09s in the
pre-tokenized comparison; 186 measured benchmark requests passed.
Audit archives preserve the earlier comparison scripts, environment records,
and benchmark results; the obsolete top-level scripts/, config/, and results/
directories were removed. Only ds41ctl and sglang_load.py are public entrypoints.
Tuning source/configs are archived in audit/performance/benchmark-source.tar.gz;
results and its checksum manifest are alongside. Temporary tuning scripts were
removed after validation. The archive can be extracted under .runs/tuning to
reproduce the benchmark using .venvs/bench and .runs/tokenizer/tokenizer.json.
No c13 GPU or service is managed by this controller.

As of 2026-10-31.
