"""Experimental MTI decoding for this deployment's DeepSeek V4.1 model.

Algorithm: https://arxiv.org/abs/2510.13940v4; reference implementation:
https://github.com/EnVision-Research/MTI (ac848621b601fef1b61807513dce3f2023bb0e82).
This is an independent SGLang port, not an upstream-supported implementation.
Requires ordinary decoding, synchronous scheduling, TP4/EP4, and paged SWA.
The negative continuation is temporary: prefix KV is shared, pending compressor
and Engram states are restored, and newly allocated pages are released.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import time

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / '.runs/mti/runtime.json'
_settings = {}
_stamp = None


def settings():
    global _settings, _stamp
    stamp = CONTROL.stat().st_mtime_ns if CONTROL.exists() else None
    if stamp != _stamp:
        value = json.loads(CONTROL.read_text()) if stamp else {}
        for key, default in [('threshold', 1.5), ('scale', 1.5)]:
            v = value.get(key, default)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f'MTI {key} must be finite')
        if value.get('threshold', 1.5) < 0 or value.get('scale', 1.5) < 1:
            raise ValueError('MTI requires threshold >= 0 and scale >= 1')
        _settings, _stamp = value, stamp
    return _settings


def entropy(logits):
    import torch
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -torch.special.xlogy(logp.exp(), logp.exp()).sum(-1)


def guide(conditional, negative, scale):
    # Combining raw logits equals combining log probabilities up to a per-row
    # additive constant, which softmax/argmax remove. Avoid two extra softmaxes.
    return scale * conditional.float() + (1.0 - scale) * negative.float()


def _capture(cfg, raw, **extra):
    # Opt-in validation of synthetic, locally constructed prompts only. Never
    # enabled by the benchmark runner or normal serving configuration.
    name = cfg.get('capture')
    if name and Path(name).name == name:
        import torch
        from sglang.srt.runtime_context import get_parallel
        if get_parallel().tp_group.rank_in_group != 0:
            return
        dest = ROOT / '.runs/mti/validation' / (name + '.pt')
        if not dest.exists():
            dest.parent.mkdir(exist_ok=True, parents=True)
            values = dict(raw=raw.detach().cpu(), **{
                k: v.detach().cpu() if hasattr(v, 'detach') else v for k, v in extra.items()})
            torch.save(values, dest)


def _validate(runner):
    from sglang.srt.runtime_context import get_parallel, get_schedule
    pool = runner.token_to_kv_pool
    if runner.model_config.hf_config.model_type != 'deepseek_v41':
        raise RuntimeError('MTI port currently supports DeepSeek V4.1 only')
    if runner.spec_algorithm.is_speculative() or not get_schedule().disable_overlap_schedule:
        raise RuntimeError('MTI requires no speculation and --disable-overlap-schedule')
    p = get_parallel()
    if p.tp_size != 4 or p.attn_dp_size != 1 or p.pp_size != 1:
        raise RuntimeError('MTI port requires TP4, attention DP1, PP1')
    if pool.request_window is not None or pool.swa_req_ring_size is not None:
        raise RuntimeError('MTI port requires paged SWA storage')
    if set(pool.compression_ratios) - {0, 1, 2}:
        raise RuntimeError('MTI state restoration is validated for ratios 0/1/2 only')
    if runner.kv_index_translator.reads_are_translated:
        raise RuntimeError('MTI port requires the static KV allocator')
    states = pool.compress_state_pools + pool.indexer_compress_state_pools
    if any(s is not None and (s.ratio != 2 or not s.request_scoped) for s in states):
        raise RuntimeError('Unsupported MTI compressor state layout')


def negative_logits(runner, batch, selected, negative_ids):
    """Evaluate the short cue without committing it to the request's state."""
    import torch
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode, ForwardBatch, ForwardMode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    pool = runner.token_to_kv_pool
    allocator = runner.token_to_kv_pool_allocator
    table = runner.req_to_token_pool.req_to_token
    rows = batch.req_pool_indices[selected].long()
    lengths = batch.seq_lens[selected].long()
    lengths_cpu = lengths.cpu()
    width, bs = len(negative_ids), len(selected)
    if int(lengths_cpu.max()) + width >= table.shape[1]:
        raise RuntimeError('MTI negative branch would exceed the context window')
    offsets = torch.arange(width, device=rows.device)
    cols = lengths[:, None] + offsets
    old_table = table[rows[:, None], cols].clone()
    last_loc = table[rows, lengths - 1].long()
    end_lengths = lengths + width
    end_cpu = lengths_cpu + width
    # The allocator reuses the uncommitted tail of the current page and allocates
    # new pages only at a page boundary, exactly as ordinary decoding does.
    locs = allocator.alloc_extend(
        lengths, lengths_cpu, end_lengths, end_cpu, last_loc, bs * width,
    )
    if locs is None:
        raise RuntimeError('MTI negative branch has insufficient KV pages')
    locs = locs.reshape(bs, width)
    # Release only pages that this branch allocated. The partly filled prefix
    # page is owned by the real request and must never be freed here.
    new_page_mask = cols >= ((lengths[:, None] + allocator.page_size - 1)
                             // allocator.page_size * allocator.page_size)
    new_page_locs = locs[new_page_mask].clone()
    state_tensors = [s.kv_score_buffer.kv_score for s in
                     pool.compress_state_pools + pool.indexer_compress_state_pools
                     if s is not None]
    hasher = getattr(getattr(runner.model, 'model', runner.model), 'engram_hasher', None)
    if hasher is not None and hasher.history is not None:
        state_tensors.append(hasher.history)
    saved_states = [(tensor, tensor.clone()) for tensor in state_tensors]
    backend = runner.attn_backend
    old_metadata = backend.forward_metadata
    old_tail = backend.tail_forward_metadata
    try:
        table[rows[:, None], cols] = locs.to(table.dtype)
        result = None
        for offset, token in enumerate(negative_ids):
            seq = (lengths + offset + 1).to(batch.seq_lens.dtype)
            seq_cpu = lengths_cpu.to(torch.int32) + offset + 1
            fb = ForwardBatch(
                forward_mode=ForwardMode.DECODE, batch_size=bs,
                input_ids=torch.full((bs,), token, dtype=torch.long, device=rows.device),
                req_pool_indices=rows.to(batch.req_pool_indices.dtype),
                seq_lens=seq, seq_lens_cpu=seq_cpu,
                req_pool_indices_cpu=rows.cpu(),
                out_cache_loc=locs[:, offset].contiguous(), seq_lens_sum=int(seq_cpu.sum()),
                positions=seq.long() - 1, orig_seq_lens=seq,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                return_logprob=False, is_prefill_only=False,
                can_run_decode_cuda_graph=True,
            )
            runner.kv_index_translator.own_plan(fb).bind(fb, runner.kv_index_translator)
            result = runner.forward(fb).logits_output.next_token_logits
        return result[:bs].clone()
    finally:
        for tensor, saved in saved_states:
            tensor.copy_(saved)
        table[rows[:, None], cols] = old_table
        backend.forward_metadata, backend.tail_forward_metadata = old_metadata, old_tail
        if new_page_locs.numel():
            allocator.free(new_page_locs)


def _validation_cache_snapshot(runner, batch):
    """Diagnostic only: snapshot reachable synthetic-prefix pages as raw bytes."""
    import torch
    pool = runner.token_to_kv_pool
    allocator = runner.token_to_kv_pool_allocator
    row, length = int(batch.req_pool_indices[0]), int(batch.seq_lens[0])
    full = runner.req_to_token_pool.req_to_token[row, :length].long().clone()
    snapshots = []

    def add(name, subpool, buffers, locs, ratio):
        pages = torch.unique(locs.long() // subpool.page_size)
        pages = pages[pages > 0]
        complete = torch.ones_like(pages, dtype=torch.bool)
        if length % (subpool.page_size * ratio) and locs.numel():
            complete &= pages != locs[-1].long() // subpool.page_size
        for i, buf in enumerate(buffers):
            snapshots.append((f'{name}/{i}', buf, pages, complete, buf[pages].clone()))

    swa = allocator.translate_loc_from_full_to_swa(full)
    swa = swa[swa > 0]
    add('swa', pool.swa_kv_pool, pool.swa_kv_pool.kv_buffer, swa, 1)
    for ratio, subpool in pool.kv_pools.items():
        if subpool is not None:
            add(f'kv{ratio}', subpool, subpool.kv_buffer, full // ratio, ratio)
    for ratio, subpool in pool.index_pools.items():
        add(f'index{ratio}', subpool, subpool.index_k_with_scale_buffer, full // ratio, ratio)

    def inspect(restore=False):
        report = {}
        for name, buf, pages, complete, saved in snapshots:
            changed = (buf[pages] != saved).reshape(len(pages), -1).any(-1)
            if changed.any():
                report[name] = dict(full_pages=int((changed & complete).sum()),
                                    partial_pages=int((changed & ~complete).sum()))
            if restore:
                buf[pages] = saved
        return report

    return inspect


def install():
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_parallel
    if getattr(ModelRunner, '_ds41_mti_installed', False):
        return
    original = ModelRunner.sample

    def sample(self, output, batch):
        cfg = settings()
        revision = cfg.get('debug_reload_revision')
        if revision and revision != getattr(ModelRunner, '_ds41_mti_revision', None):
            # Local experiment debugging only, changed at idle. Normal serving
            # never loads this adapter or accepts this control through an API.
            import importlib
            import sys
            module = importlib.reload(sys.modules[__name__])
            ModelRunner.sample = original
            ModelRunner._ds41_mti_installed = False
            module.install()
            ModelRunner._ds41_mti_revision = revision
            return ModelRunner.sample(self, output, batch)
        # Keep synthetic validation (including forced reference tokens) scoped
        # to its exact request. Other clients continue through native sampling.
        validation_rid = cfg.get('validation_rid')
        if validation_rid and batch.rids != [validation_rid]:
            return original(self, output, batch)
        force_ids = cfg.get('validation_force_ids')
        if force_ids is not None:
            # Synthetic validation only: independently advance the native
            # scheduler through the cue and capture its final decode logits.
            # No MTI branch is executed on this reference path.
            if batch.batch_size != 1 or cfg.get('enabled', False):
                raise RuntimeError('Forced-cue validation requires one request and MTI disabled')
            name = cfg.get('capture')
            if getattr(self, '_mti_force_name', None) != name:
                self._mti_force_name, self._mti_force_step = name, 0
            step = self._mti_force_step
            if step == 0 and cfg.get('validation_disable_graph', False):
                self._mti_validation_graph = self.decode_cuda_graph_runner
                self.decode_cuda_graph_runner = None
            if step == 0 and cfg.get('validation_stable_topk', False):
                from mti_validation import install_stable_topk
                self._mti_stable_topk_restore = install_stable_topk()
            if step == 0 and cfg.get('validation_sort_indices', False):
                cls = type(self.attn_backend)
                native_topk = cls._low_ratio_index_topk
                self._mti_validation_topk = cls, native_topk
                self._mti_validation_trace = []
                def sorted_topk(backend, layer, *args, **kwargs):
                    trace = None
                    if cfg.get('validation_trace', False):
                        trace = dict(stage=getattr(self, '_mti_trace_stage', 'native'),
                                     layer=layer.layer_id, position=int(args[3][0]),
                                     x=args[0].detach().cpu().clone(),
                                     q_lora=args[1].detach().cpu().clone())
                    from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk
                    transform = full_topk.topk_transform_paged_from_metadata
                    def traced_transform(logits, metadata, *a, **kw):
                        if trace is not None:
                            trace['scores'] = logits.detach().cpu().clone()
                            trace['score_lens'] = metadata.compressed_seq_lens.detach().cpu().clone()
                        return transform(logits, metadata, *a, **kw)
                    if trace is not None:
                        full_topk.topk_transform_paged_from_metadata = traced_transform
                    try:
                        result = native_topk(backend, layer, *args, **kwargs)
                    finally:
                        full_topk.topk_transform_paged_from_metadata = transform
                    indices = backend.forward_metadata.core_metadata.sparse_page_indices(layer.compress_ratio)
                    sentinel = torch.iinfo(indices.dtype).max
                    ordered = indices.masked_fill(indices < 0, sentinel).sort(-1).values
                    indices.copy_(ordered.masked_fill(ordered == sentinel, -1))
                    if trace is not None:
                        trace['indices'] = indices.detach().cpu().clone()
                        self._mti_validation_trace.append(trace)
                    return result
                cls._low_ratio_index_topk = sorted_topk
            raw = output.next_token_logits
            if step == 0:
                saved = raw.clone()
                extra = {}
                if cfg.get('validation_paired_branch', False):
                    # Compare against the scheduler continuing this very same
                    # request, avoiding independently recomputed prefix KV.
                    _validate(self)
                    selected = torch.zeros(1, dtype=torch.long, device=batch.seq_lens.device)
                    if cfg.get('validation_sync', False):
                        torch.cuda.synchronize()
                    inspect = (_validation_cache_snapshot(self, batch)
                               if cfg.get('validation_cache_check', False) else None)
                    self._mti_trace_stage = 'aux1'
                    extra['negative'] = negative_logits(self, batch, selected, force_ids)
                    if cfg.get('validation_sync', False):
                        torch.cuda.synchronize()
                    if inspect:
                        extra['cache_changes_first'] = inspect(cfg.get('validation_restore_cache', False))
                    self._mti_trace_stage = 'aux2'
                    extra['negative_repeat'] = negative_logits(self, batch, selected, force_ids)
                    self._mti_trace_stage = 'native'
                    if cfg.get('validation_sync', False):
                        torch.cuda.synchronize()
                    if inspect:
                        extra['cache_changes_second'] = inspect(cfg.get('validation_restore_cache', False))
                    output.next_token_logits = saved
                _capture(dict(cfg, capture=name + '-initial'), saved,
                         lengths=batch.seq_lens, rows=batch.req_pool_indices, **extra)
            if step == len(force_ids):
                _capture(cfg, output.next_token_logits,
                         lengths=batch.seq_lens, rows=batch.req_pool_indices,
                         trace=getattr(self, '_mti_validation_trace', []))
                if hasattr(self, '_mti_validation_graph'):
                    self.decode_cuda_graph_runner = self._mti_validation_graph
                    del self._mti_validation_graph
                if hasattr(self, '_mti_validation_topk'):
                    cls, native_topk = self._mti_validation_topk
                    cls._low_ratio_index_topk = native_topk
                    del self._mti_validation_topk
                if hasattr(self, '_mti_stable_topk_restore'):
                    self._mti_stable_topk_restore()
                    del self._mti_stable_topk_restore
            try:
                result = original(self, output, batch)
            finally:
                output.next_token_logits = raw
            if step < len(force_ids):
                result.fill_(int(force_ids[step]))
            self._mti_force_step += 1
            return result
        if not cfg.get('enabled', False) or self.is_draft_worker or batch.is_prefill_only:
            if cfg.get('capture'):
                _capture(cfg, output.next_token_logits)
            return original(self, output, batch)
        if not getattr(self, '_mti_validated', False):
            _validate(self)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path,
                                                      trust_remote_code=False)
            self._mti_negative_ids = tokenizer.encode('OUTPUT ERROR', add_special_tokens=False)
            self._mti_validated = True
            self._mti_counts = dict(tokens=0, interventions=0, changed_top1=0,
                                    negative_forwards=0, branch_seconds=0.0)
            self._mti_log_at = 0.0
            LOG.info('MTI initialized: negative cue has %d tokens; prefix KV reused',
                     len(self._mti_negative_ids))
        if getattr(self, '_mti_run_id', None) != cfg.get('run_id', 'validation'):
            self._mti_run_id = cfg.get('run_id', 'validation')
            self._mti_counts = dict(tokens=0, interventions=0, changed_top1=0,
                                    negative_forwards=0, branch_seconds=0.0)
            self._mti_log_at = 0.0
        raw = output.next_token_logits
        mask = (entropy(raw) > cfg.get('threshold', 1.5)).to(torch.int32)
        # Every EP rank must execute the same extra model forwards, including
        # when floating-point ties around the threshold differ between ranks.
        group = get_parallel().tp_group
        group.broadcast(mask, src=0)
        selected = mask.nonzero().flatten()
        counts = self._mti_counts
        counts['tokens'] += raw.shape[0]
        if selected.numel():
            saved = raw.clone()  # decode graphs reuse the output buffer
            t0 = time.monotonic()
            neg = negative_logits(self, batch, selected, self._mti_negative_ids)
            _capture(cfg, saved, negative=neg, selected=selected,
                     lengths=batch.seq_lens, rows=batch.req_pool_indices,
                     negative_ids=self._mti_negative_ids)
            if cfg.get('shadow', False):
                combined = saved[selected]
            else:
                combined = guide(saved[selected], neg, cfg.get('scale', 1.5))
            counts['changed_top1'] += int((combined.argmax(-1) != saved[selected].argmax(-1)).sum())
            saved[selected] = combined.to(saved.dtype)
            output.next_token_logits = saved
            counts['interventions'] += selected.numel()
            counts['negative_forwards'] += len(self._mti_negative_ids)
            counts['branch_seconds'] += time.monotonic() - t0
        if group.rank_in_group == 0 and time.monotonic() - self._mti_log_at > 30:
            LOG.info('MTI counters: %s', json.dumps(counts, sort_keys=True))
            self._mti_log_at = time.monotonic()
            destination = ROOT / '.runs/mti/counters.json'
            temp = destination.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(counts, threshold=cfg.get('threshold', 1.5),
                                           scale=cfg.get('scale', 1.5), shadow=cfg.get('shadow', False),
                                           run_id=self._mti_run_id)))
            temp.replace(destination)
        try:
            return original(self, output, batch)
        finally:
            # Graph runners may reuse the LogitsProcessorOutput object itself.
            # Keep its original captured buffer attached for the next replay.
            output.next_token_logits = raw

    ModelRunner.sample = sample
    ModelRunner._ds41_mti_installed = True
