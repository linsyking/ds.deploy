# Combining MTI with DSpark

Assessment only; no combined decoder has been implemented or benchmarked.

The previous Maxwell trial measured DSpark at 5/5 correct, 1,425 seconds and
1,576,410 reported output tokens. Ordinary MTI achieved 5/5, 3,242 seconds and
872,170 output tokens. These counts include reasoning and visible output;
separate thinking-token totals were unavailable. See `MTI_EXPERIMENT.md`.

If MTI's approximately 45% token reduction relative to DSpark held in a combined
decoder, it could tolerate approximately 1.8 times DSpark's time per generated
token before losing the total-time advantage. This is a break-even estimate,
not a prediction of combined performance.

## Integration

DSpark should generate proposals, while MTI defines the target distribution
used to verify those proposals. Insert guidance after raw target verification
logits are computed, before grammar masks, penalties, temperature, sampling and
acceptance. Below the entropy threshold use ordinary target logits; above it
use `1.5 * conditional_logits - 0.5 * negative_logits`.

For greedy decoding, compare proposals with guided target argmaxes. For sampled
decoding, acceptance and residual sampling must use the guided target distribution
and the actual draft proposal distribution. Initially the draft can remain
unchanged. Guiding only the draft leaves the original target distribution after
correct verification; guiding already accepted tokens is too late. Guiding only
the bonus token is a different algorithm that misses interventions inside blocks.

The current `mti.py` hooks ordinary `ModelRunner.sample` and explicitly rejects
speculation and overlap. It cannot be enabled alongside DSpark without changes.
Relevant installed source files are:

- `sglang/srt/speculative/dspark_components/dspark_verify.py`: raw verification
  sampling adjustments near lines 337 and 526, graph acceptance near 749,
  separate target/proposal inputs near 1246.
- `sglang/srt/speculative/dspark_components/dspark_worker_v2.py`: folded acceptance
  and state commits near 1120.
- `sglang/srt/layers/engram.py`: history commits near 374.

Paths and line numbers refer to this deployment's installed SGLang 0.5.23.

## Smallest prototype

Use synchronous DSpark for one request, with overlap disabled, eager classic
verification, simulated acceptance disabled, and both folded graph acceptance
paths disabled. At the first high-entropy position, stop the speculative chain,
emit one MTI-guided token, discard the unused speculative suffix, and resume
DSpark. Prefix proposals must still pass the appropriate native verifier.

The auxiliary negative branch needs the exact candidate prefix, matching
ratio-2 compressor and Engram histories, isolated temporary KV, and restored
request tables, attention metadata, logits and hidden states. DSpark has already
populated a speculative suffix, so the ordinary MTI rollback helper is not
sufficient unchanged. Engram commits must agree with the accepted prefix.

Validate fixed-logit greedy behavior, sampled distributions, acceptance and
bonus boundaries, grammar/budget enforcement, and state restoration before any
Maxwell timing run. Equal RNG seeds do not imply identical speculative and
ordinary samples; native BF16 sparse-attention ties also limit bitwise equality.

Then batch negative branches and investigate a single forward for the three-token
cue. Restoring production overlap and graph acceptance requires explicit state
ownership and stream synchronization. The present three-forward branch can
erase the speculative speed advantage; its cost and accepted tokens per forward
must be measured in addition to correctness, total tokens and wall time.

## Literature and official-code review — 2026-11-03

This section is research only. No combined decoder, new model request or GPU
benchmark was run for this review. Published speedups below use each paper's
own models, hardware and baselines; none demonstrates a speedup over this
deployment's DSpark configuration on Maxwell.

The strongest precedent is **SpecExit**, which explicitly combines a speculative
decoder with a cheap learned decision to end thinking. **DoubtLess** is the most
relevant public implementation of training-free logit steering with a small
helper. **SpecReason** directly demonstrates composition between reasoning-level
and token-level speculation. These are different ways to change the reasoning
policy; preserving ordinary DSpark's original output distribution cannot itself
systematically shorten the expected reasoning trace. DSpark's own paper reports
60–85% faster generation than MTP-1 at matched throughput in DeepSeek-V4 serving,
not shorter thinking. [DSpark paper](https://arxiv.org/abs/2607.05147)

### Candidates and what their results establish

| Method | Published evidence and actual baseline | Applicability here |
| --- | --- | --- |
| **SpecExit** | Trained confidence/progress/remaining-length signals select paragraph boundaries for inserting the thinking-end token. On R1-distill-Llama-8B GSM8K, EAGLE3 → SpecExit changes 976 → 333 tokens and 276.9 → 112.6 seconds, **but accuracy falls 79.3% → 75.3%**. Experiments use vLLM and 8×H20. | Closest architecture for retaining DSpark's fast decoding while shortening traces. Needs V4.1-specific features, labels and training. Its gains are neither universal nor quality guarantees: Qwen3-4B AIME is slower than EAGLE3, 206.1 vs 187.3 seconds. [Paper, Table 1](https://arxiv.org/html/2509.24248v2#S4.T1) |
| **DoubtLess** | A small model supplies brief-minus-verbose logits during thinking. Reports 31% less thinking with Qwen3-32B/Qwen3-0.6B; 20% less per turn in a 164-problem HumanEval+ agent with one repair turn. A 14-problem, single-B200 timing sample improves 156.2 → 119.4 seconds per problem over ordinary decoding. | Relevant training-free alternative to MTI's expensive auxiliary target forwards. Published timing does **not** include DSpark. Needs a compatible small model and a guided verifier port. “Unchanged accuracy” describes aggregate experimental evidence, not identical outcomes or a per-task guarantee. [Paper, §§4.3, 5.3](https://arxiv.org/html/2610.18862v1) |
| **SpecReason** | The small reasoner proposes a paragraph; the large reasoner scores it and regenerates rejected steps. QwQ/Skywork-32B with either of two 1.5B helpers, vLLM 0.8.2, two A6000s. Composition reduces latency 8.8–58.0% relative to a five-token speculative-decoding baseline across model pairs/datasets in the inspected v2 experiments. | Demonstrates that the two acceleration levels can compose. Requires an independent small reasoner and semantic acceptance, changing the target trajectory. Math/science evaluation does not establish coding-agent quality. [Paper, §§4–5](https://arxiv.org/html/2504.07891v2) |
| **SeedThink** | Difficulty-aware seed thoughts shorten subsequent reasoning and also serve as a model-free speculative draft corpus. | Directly matches the dual objective conceptually. Only the indexed primary abstract was verifiable; the full OpenReview paper requires browser verification, and no official implementation was identified. Treat as an unverified implementation lead, not a ready port. [Primary paper](https://openreview.net/forum?id=cy7YVhpW4u) |
| **R-Stitch / R-Stitch+** | Entropy-based routing lets a small model emit easy tokens and delegates uncertain tokens to the large model; the plus variant trains a router. Latest v6 reports peak 3.00–4.10× speedups over full 7B/14B/32B decoding on math, with shorter traces, and includes EAGLE3 comparisons. | Changes the large model's token distribution. Main results use batch size 1; Appendix A.8 shows gains shrinking and reversing at larger batches. Coding speedups shrink when preserving accuracy. The official repository still contains README/assets and promises code later. [Latest paper](https://arxiv.org/html/2507.17307v6), [repository](https://github.com/Caesarhhh/R_Stitch) |
| **DEER** | At transition points, probes an answer and stops reasoning when answer confidence is high. Reports token savings on several reasoning families. | Easy conceptual compatibility with DSpark at the request/controller layer, but probes add work and confidence needs task calibration. The public repository's coding implementation and branch-parallel acceleration remain marked forthcoming. [Paper](https://arxiv.org/abs/2504.15895), [official code](https://github.com/iie-ycx/DEER) |
| **SpecGen** | Forks non-reasoning code candidates from an ongoing reasoning prefix; actual kernel validation/profiling can terminate the long branch. Reports 1.68–1.82× faster end-to-end optimization than CudaForge/AlphaEvolve/KernelAgent on H200 with GLM-5.1 and **DeepSeek-V4-Pro**. | Particularly relevant agent precedent. Orthogonal to token speculation, but needs cheap objective online validation and spare concurrency. It does not prove fewer *total* tokens across all forks. No released official code was found in the paper or author project search. [Paper](https://arxiv.org/html/2606.17518v1), [author page](https://yezhisheng.me/publication/specgen/) |

### What the official implementations actually contain

**SpecExit:** inspected AngelSlim commit
`44a066419a5d6293143092dfb08b3147e0b50ac2` and its official vLLM PR.
`Eagle3Model.eagenerate` smooths signals, finds a split in the accepted block,
truncates the accepted length, then substitutes the thinking-end recovery token
before updating inference state. The drafter extracts signal dimensions from
its feature projection. This is a controller at a valid block boundary, not
MTI-style guidance applied to already committed tokens.
[Controller, lines 469–574](https://github.com/Tencent/AngelSlim/blob/44a066419a5d6293143092dfb08b3147e0b50ac2/angelslim/compressor/speculative/inference/models/eagle3/eagle3_model.py#L469),
[signal extraction, lines 628–646](https://github.com/Tencent/AngelSlim/blob/44a066419a5d6293143092dfb08b3147e0b50ac2/angelslim/compressor/speculative/inference/models/eagle3/draft/llama3_eagle3.py#L628).

The official [vLLM PR #27192](https://github.com/vllm-project/vllm/pull/27192)
targets EAGLE3 Qwen/Llama implementations and was closed unmerged on
2026-02-18. It supplies useful integration code, not a supported SGLang/DSpark
configuration. A trained exit head from a different model cannot be transferred
to V4.1 without validation. The paper constructs supervision by testing earlier
reasoning cut points; collecting equivalent labels for successful tool-using
trajectories would be a substantial part of the work.
[Training method](https://arxiv.org/html/2509.24248v2#S3.SS2)

**DoubtLess:** inspected commit
`4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372`.
The optimized implementation runs brief/verbose helper passes as a doubled
batch on a CUDA stream alongside the base model, with separate instruction KV.
It then combines the logits, restricts steering to the base nucleus and
preserves EOS/thinking-close logits. Preserving those logits does not preserve
their normalized probabilities. The small helper requires matching token IDs;
its optimized decoder supports full-attention Qwen3/Llama/Qwen2. An SGLang port
needs helper caches for every candidate prefix and rollback after rejection.
Our DSpark head is not an independent brief/verbose reasoner, and no suitable
standalone V4.1-tokenizer helper was identified in this review.
[Implementation, lines 354–462](https://github.com/leoluolol/DoubtLess/blob/4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372/doubtless/vllm/model.py#L354),
[contrast rule](https://github.com/leoluolol/DoubtLess/blob/4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372/doubtless/contrast.py#L28),
[supported backends](https://github.com/leoluolol/DoubtLess#backends).

**SpecReason:** inspected commit
`6fd5b12929590e6824957191cc2e40a93d021339`.
The public proof of concept uses two API servers, `\n\n` step boundaries,
one-token score generation, and sequential fallback. It assumes continuation
of an unfinished assistant thought and math-style answer markers. Its
`final_num_output_tokens` budget counts the chosen step, excluding rejected
helper output and score tokens; therefore this counter is not total inference
work. Keeping DSpark in the large server is plausible, but adapting the
controller to native tool messages and long-context agent state is additional
work. [Code, lines 79–144 and 257–311](https://github.com/ruipeterpan/specreason/blob/6fd5b12929590e6824957191cc2e40a93d021339/spec_reason.py#L79),
[deployment instructions](https://github.com/ruipeterpan/specreason#launching-vllm-servers).

**DEER:** inspected commit `c9dd19fbffa27f841cfe47502d015b63811b4d1b`.
The available vLLM script alternates thinking chunks with confidence probes and
then an answer phase; it is not a drop-in decoder patch. Its math answer
inducer and confidence calculation must not be assumed to work for arbitrary
agent turns. [Probe/decision code](https://github.com/iie-ycx/DEER/blob/c9dd19fbffa27f841cfe47502d015b63811b4d1b/vllm-deer.py#L624).

### Conditional recommendation for this deployment

**Best architectural fit:** a SpecExit-inspired tiny exit head evaluated from
already available target/draft features at accepted paragraph boundaries.
This could preserve nearly all DSpark decoding speed by avoiding MTI's extra
large-model forwards. That is an engineering hypothesis, not a measurement.
It requires feature collection, independently held-out successful trajectories,
cut-point labeling, head training/calibration, and a controller before KV,
compressor, Engram, grammar and output-state commits. A first version could use
one request and overlap off, then restore fused/graph paths after correctness
validation. Forcing the thinking-close marker changes the reasoning policy;
ordinary DSpark's lossless guarantee does not extend to the intervention.

**Best training-free route using assets already here:** the MTI-guided verifier
prototype described above. Its advantage is an existing V4.1-compatible
intervention, but its full-target auxiliary calls create a larger latency risk
than a tiny controller. DoubtLess becomes a serious alternative if a compatible
small helper is available; using a different tokenizer needs a different
algorithm, not a direct logit subtraction. SpecReason and SpecGen are larger
harness projects. SpecGen is attractive only where genuine online validation
is available; a benchmark's hidden final correctness check cannot serve as a
runtime early-stop oracle.

**Do not optimize acceptance rate alone.** A recent primary study of twelve
target/drafter pairs finds relaxed verification accepts 1.34× more tokens per
round but produces 1.54× longer completions at its most relaxed tested settings,
often erasing the apparent gain. This argues against assuming permissive
DSpark acceptance will shorten thinking.
[Acceptance Is Not Speed, 2026-10-16](https://arxiv.org/abs/2610.23818)

For guided sampling, retain the actual proposal distribution and verify against
the guided target distribution. Preserve DSpark's non-anticipating scheduling:
admitting token k must not depend on seeing token k or a future candidate.
Re-ranking whole candidate blocks by predicted final brevity can violate this
condition. A learned early-exit policy must be documented as an intentional
policy change instead of claimed as lossless speculation.
[DSpark, §3.2.2 and Appendix A](https://arxiv.org/html/2607.05147v1)

Before selecting an implementation, compare candidates under the same model,
effort, output budget, compaction setting and concurrency. Use Maxwell answer
correctness and total benchmark wall time as the primary criteria, including
all probes, rejected drafts, tool work and compactions in the timing. Separately
count committed thinking tokens, visible output, discarded auxiliary tokens,
acceptance length, target/helper forward time and tail latency. Repeat enough
seeds/tasks to expose quality regressions; the prior single 5/5 trial cannot
establish equivalent quality. Published speedups should not be multiplied by
our DSpark speedup to predict a combined result.
