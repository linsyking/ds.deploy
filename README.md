# DeepSeek V4.1 Flash on c02

Single-node deployment of deepseek-ai/DeepSeek-V4.1-Flash-1017 on c02
(10.78.202.30), using four GB200 GPUs and SGLang 0.5.23.
Checkpoint revision: 1bae0e1088993b9b82de08f59de5be5a34e42170.
The downloaded checkpoint is hash-verified; native vision and DSpark draft
weights are included. The model path and tunable values are in serving.json.

## Operations

Run these commands in this repository on c02:

    ./ds41ctl start
    ./ds41ctl status
    ./ds41ctl logs --follow
    ./ds41ctl restart
    ./ds41ctl stop
    ./ds41ctl load
    ./sglang_load.py 2
    ./ds41ctl check
    ./ds41ctl check-reasoning
    ./ds41ctl check-budget
    ./ds41ctl check-search

The active target is now c02. Local start/restart uses local files; sync is a
no-op on c02. The former c04 service was stopped and is being left to exit after
a driver teardown stall. Its disk files were used as copy sources only.

Remote start/restart sync the small runtime files to the configured target. The engine runs in a named
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

Internal OpenAI-compatible base URL: http://10.78.202.30:18000/v1
Anthropic-compatible base URL: http://10.78.202.30:18000
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
left thinking at the engine's off default. The engine package files are unmodified; thinking_budget.py installs a small runtime adapter for explicit token budgets.

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
implemented as enabled thinking, not Claude's adaptive policy. The staged
thinking_budget.py adapter connects Anthropic thinking.budget_tokens to native
SGLang grammar masks when enforce_thinking_budget=true. max_tokens still limits
reasoning plus answer together. Live budget validation on c02 is pending while
checkpoint verification and first startup complete.

Core text, reasoning, tool calls/round trips, images and streaming have been checked.
This does not certify every OpenAI/Anthropic endpoint or provider-specific feature;
Responses and cloud-service APIs are outside this compatibility test scope.
One random streaming-image content assertion failed without a retained response;
retesting a known fixture through OpenAI streaming/non-streaming and Anthropic
streaming succeeded. Protocol compatibility does not guarantee recognition accuracy.
Details are in audit/reasoning-check-20261031-154801/.

DSH's Anthropic provider has two different modes:

- Budget mode (forceAdaptiveThinking=false or omitted): sends
  thinking.type=enabled, thinking.budget_tokens, and max_tokens; it omits
  output_config.effort. In the inspected pi-ai 1.3.0 transport, low=2048,
  medium=8192, high=16384, max=16384 thinking tokens. These defaults can be
  adjusted or clamped by the client. Off sends thinking.type=disabled.
- Adaptive mode (forceAdaptiveThinking=true): sends thinking.type=adaptive
  and output_config.effort=low/medium/high/max, without budget_tokens.

To use the numerical budgets, remove the earlier forceAdaptiveThinking=true
setting or set it to false in the existing model's compat section. This is a
client setting; no client files have been changed here. Explicit effort, if also
supplied, is preserved. Budget-only requests retain the server's max effort
and differ by the enforced token cap, not by an inferred effort label.

The transport may add its thinking allowance to its base answer allowance:
with a 4096 base allowance, medium sent max_tokens=12288 and budget_tokens=8192;
high/max sent max_tokens=20480 and budget_tokens=16384. The actual max_tokens
field, not the UI label, remains the total output ceiling.

A local reproduction using the pi-ai 1.3.0 dependency pinned by current DSH,
including its upstream DSH patch, verified both request fields and streamed
output against c04. With adaptive mode, off returned zero reasoning characters,
medium returned 1050, and max returned 2408 on one synthetic prompt. These are
diagnostics, not latency or quality guarantees. The user's inaccessible DSH
installation and its reported identical 39-second times have not been inspected;
the off failure did not reproduce in the current transport. Evidence is in
audit/dsh-reasoning-20261031-155906/.

Claude Code's "response exceeded ... output token maximum" message can occur
even when CLAUDE_CODE_MAX_OUTPUT_TOKENS is honored: it reports an API response
ending with stop_reason=max_tokens. Reasoning and the final answer share that
limit. An actual Claude Code 2.1.314 test sent max_tokens=256, exhausted exactly
256 tokens on each attempt, and reproduced that error; a second test sent
max_tokens=16384 and completed normally after 426 tokens. These tests used
isolated settings and a temporary loopback relay to capture only limit fields
and response metadata. Evidence: audit/claude-budget-20261031-181030/.

In this tested version, MAX_THINKING_TOKENS=1024 did not send a token budget for
the custom DeepSeek model: Claude sent adaptive thinking and effort=high.
CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1 also retained adaptive mode for this model
in a follow-up test. Do not assume either setting supplies a hard budget here.
The staged adapter addresses explicit numerical budgets. Adaptive requests
carry no numerical budget and retain the current default policy. No automatic
cap is inferred from an effort label.

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

## PDF and text documents

`enable_documents=true` registers `documents.py` in the SGLang API process.
It accepts Anthropic `document` blocks on `/v1/messages` and
`/v1/messages/count_tokens`, both as user attachments and inside
`tool_result.content` (including Claude Code's Read tool). Before native
Pydantic validation, each PDF page becomes a text block with its extracted
text and page number, followed by an image block containing that same page.
SGLang's native multimodal converter and DeepSeek vision encoder process both.
Image-only/scanned pages retain their page image. This follows Anthropic's
[text plus page image design](https://platform.claude.com/docs/en/build-with-claude/pdf-support).

Supported sources are base64 `application/pdf`, HTTP(S) PDF URLs,
base64 UTF-8 `text/plain`, and `source: {"type":"text","data":"..."}`.
Document `title` and `context` are included with the page text. Files API
`file_id` references and native citation events are not implemented; requests
enabling citations return a clear 400. Existing native response/SSE formats
are unchanged. Invalid/encrypted PDFs are rejected rather than silently
replaced with empty text. No client, API key, forwarding or model change is
required. Disable the adapter with `enable_documents=false` and restart.

PDFium renders at 144 DPI, capped at 2048 pixels per edge and 4 million pixels
per page, with JPEG quality 92. Limits are 128 MiB of incoming JSON, 96 MiB of
combined decoded document data, 256 MiB of expanded JSON, and 600 PDF pages
per request (100 if configured context is below 1M). Pages are never silently
truncated. The native model context limit still applies; 600 pages need not
fit. Text-heavy pages above 200,000 characters must be split. These are local
operational limits, not a promise of identical Anthropic token accounting.

Rendering runs off the event loop, with serialized PDFium access and at most
two outstanding workers. A 128 MiB, ten-minute memory cache keyed by PDF bytes
avoids repeat rendering in conversation history, including concurrent
duplicate requests. No PDF content is logged or persisted by the adapter.
URL downloads have size and time bounds. The existing bearer authentication
is checked before processing documents.

When combined with Exa, search-query extraction runs before document expansion;
the search provider receives the original user query, not extracted page text.
`count_tokens` retains SGLang's native estimate, which undercounts vision
tokens; generation `usage.input_tokens` includes the actual image expansion.

Install the additional CPU dependency in the serving environment:

    uv pip install --python .venvs/sglang/bin/python -r requirements-documents.txt
    .venvs/sglang/bin/python -m unittest -v test_documents.py test_web_search.py
    .venvs/sglang/bin/python check_documents.py

`test_documents_native.py` additionally checks the installed SGLang validator
and converter in the deployment's CUDA environment without loading the model.
The live check uses a generated two-page PDF, with an image-only code and bar
chart on page 2, and saves results under `audit/documents-*`.

## Web search

Anthropic search is enabled by enable_web_search=true in serving.json.
The Exa credential is stored separately in .secrets/exa-api-key (mode 0600),
loaded into EXA_API_KEY for the engine, and redacted from controller output.
Restart with ./ds41ctl restart after changing the setting or credential.
No client URL or API authentication change is needed, including through port 8312.

web_search.py adds a local compatibility adapter for /v1/messages. A request
declaring a web_search_<date> server tool triggers one Exa Auto search using
the last user text (or Claude Code's "for the query:" text). It supplies up to
five source excerpts to the model and removes the server search tool before
native inference. Images and ordinary client tools are preserved. Identical
queries and domain filters are cached for ten minutes, including concurrent
requests. Search failures return an explicit 502 rather than an unsearched answer.
Requests without a server search tool pass through unchanged.

The adapter reuses SGLang 0.5.23's native Exa HTTP client. Upstream Exa execution
is officially supported for GPT-OSS/OpenAI Responses; the DeepSeek Anthropic
bridge is local code, following the GLM deployment's approach. It returns
ordinary model text with source links, not native Anthropic server-tool events
or citation blocks. Dynamic filtering and autonomous multi-search tool loops
are not implemented. allowed_domains and blocked_domains are translated to
Exa filters; specifying both is rejected. Versioned search tools all use this
same basic behavior. This does not add native search to DeepSeek /v1/responses.

Use ./ds41ctl check-search for JSON/streaming search checks; this spends one
search request if the query is not cached. Unit checks require no external calls:
python3 -m unittest -v test_web_search.py.

## Remote tunnel

    ./ds41ctl tunnel-start
    ./ds41ctl tunnel-status
    ./ds41ctl tunnel-stop

FRPC runs on c02 through the loopback GOST HTTP proxy at 127.0.0.1:18080.
The route is server port 8312 -> c02 10.78.202.30:18000. c04 is not an
intermediate hop. The old c04 FRPC client has been stopped to release port 8312.

On ssh.linsyking.com, use:
OpenAI base URL: http://127.0.0.1:8312/v1
Anthropic base URL: http://127.0.0.1:8312

Both APIs require the existing API key in .secrets/api-key, sent as
Authorization: Bearer <key>. The API key is unchanged from c04. The separate
FRP authentication token and server settings remain in .secrets/frpc.toml.
Private files have mode 0600 inside a mode-0700 directory and are ignored by Git.
Credentials are not included in normal runtime sync; provision them separately.

The controller checks that FRPC port 8312 targets the configured SGLang API,
verifies API authentication, and starts GOST when needed. GOST uses Cloudflare
DNS over HTTPS. FRPC transport TLS is enabled. FRPC runs under nohup in its own
process group and reconnects after connection loss; logs are in .runs/frpc.log.
tunnel-stop stops only owned FRPC; the loopback GOST proxy remains available.
Access through server loopback is the intended scope; no firewall rules change.

FRPC 0.71.1 linux/arm64 was installed on c02 from the official release and
verified against its published SHA-256 checksum. GOST 3.3.0 was already installed.
Direct internal access remains available at http://10.78.202.30:18000/v1 for
OpenAI or http://10.78.202.30:18000 for Anthropic.

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
