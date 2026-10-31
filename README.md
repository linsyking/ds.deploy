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
DeepSeek codec and auto parsers. Reasoning remains client-controlled.

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
