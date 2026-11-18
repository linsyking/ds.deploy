"""Algorithm and KV ownership tests; run in the deployment's Python environment."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import time
import torch

from mti import entropy, guide, negative_logits


class MTITest(unittest.TestCase):
    def test_stable_validation_topk_masks_padding_and_restores_native_functions(self):
        from mti_validation import install_stable_topk
        from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk, sparse_table
        original = full_topk.topk_transform_paged_from_metadata
        restore = install_stable_topk()
        try:
            logits = torch.tensor([[0., 2., 2., 1., 100.]])
            out = torch.empty(1, 3, dtype=torch.int32)
            raw = torch.empty_like(out)
            metadata = NS(compressed_seq_lens=torch.tensor([4]),
                          page_table=torch.tensor([[10, 11, 12]]), compressed_page_size=2)
            full_topk.topk_transform_paged_from_metadata(logits, metadata, out, raw)
            self.assertEqual(out.tolist(), [[21, 22, 23]])
            self.assertEqual(raw.tolist(), [[1, 2, 3]])
            sparse_table.topk_transform_sparse(
                torch.tensor([[.2, .8, .8, .1]]), torch.tensor([4]),
                torch.tensor([[5]]), out)
            self.assertEqual(out.tolist(), [[41, 42, 40]])
        finally:
            restore()
        self.assertIs(full_topk.topk_transform_paged_from_metadata, original)

    def test_paired_reference_preserves_positive_buffer_before_forcing_cue(self):
        import mti
        from sglang.srt.model_executor.model_runner import ModelRunner
        original = ModelRunner.sample
        raw = torch.tensor([[1., 2., 3.]])
        output = NS(next_token_logits=raw)
        seen = []
        def native(self, output, batch):
            seen.append(output.next_token_logits.clone())
            return output.next_token_logits.argmax(-1)
        def negative(*args):
            raw.fill_(999.)  # Simulate a replay overwriting its captured buffer.
            return torch.zeros_like(raw)
        try:
            ModelRunner.sample = native
            with patch('mti.settings', return_value=dict(
                    enabled=False, validation_rid='synthetic', capture='synthetic',
                    validation_force_ids=[10, 11], validation_paired_branch=True)), \
                 patch('mti._validate'), patch('mti._capture'), \
                 patch('mti.negative_logits', side_effect=negative) as branch:
                mti.install()
                result = ModelRunner.sample(NS(), output, NS(
                    rids=['synthetic'], batch_size=1, seq_lens=torch.tensor([128]),
                    req_pool_indices=torch.tensor([0])))
            self.assertEqual(result.tolist(), [10])
            self.assertEqual(branch.call_count, 2)
            torch.testing.assert_close(seen[0], torch.tensor([[1., 2., 3.]]))
            self.assertIs(output.next_token_logits, raw)
        finally:
            ModelRunner.sample = original
            if hasattr(ModelRunner, '_ds41_mti_installed'):
                del ModelRunner._ds41_mti_installed

    def test_forced_validation_does_not_change_another_request(self):
        import mti
        from sglang.srt.model_executor.model_runner import ModelRunner
        original = ModelRunner.sample
        native = Mock(return_value=torch.tensor([9]))
        runner = NS()
        try:
            ModelRunner.sample = native
            with patch('mti.settings', return_value=dict(
                    enabled=False, validation_rid='synthetic',
                    validation_force_ids=[10, 11], capture='synthetic')):
                mti.install()
                for rids in [['another-client'], ['synthetic', 'another-client']]:
                    result = ModelRunner.sample(runner, NS(), NS(rids=rids))
                    self.assertEqual(result.tolist(), [9])
            self.assertEqual(native.call_count, 2)
            self.assertFalse(hasattr(runner, '_mti_force_step'))
        finally:
            ModelRunner.sample = original
            if hasattr(ModelRunner, '_ds41_mti_installed'):
                del ModelRunner._ds41_mti_installed

    def test_sampling_restores_captured_output_buffer(self):
        import mti
        from sglang.srt.model_executor.model_runner import ModelRunner
        original = ModelRunner.sample
        raw = torch.tensor([[1., 2., 3.]])
        output = NS(next_token_logits=raw)
        runner = NS(is_draft_worker=False, _mti_validated=True,
                    _mti_negative_ids=[10, 11], _mti_run_id='validation',
                    _mti_log_at=time.monotonic(), _mti_counts=dict(
                        tokens=0, interventions=0, changed_top1=0, negative_forwards=0, branch_seconds=0.))
        def native_sample(self, output, batch):
            return output.next_token_logits.argmax(-1)
        group = NS(rank_in_group=1, broadcast=lambda *args, **kwargs: None)
        try:
            ModelRunner.sample = native_sample
            with patch('mti.settings', return_value=dict(enabled=True, threshold=0, scale=1.5)), \
                 patch('mti.negative_logits', return_value=torch.tensor([[0., 0., 10.]])), \
                 patch('sglang.srt.runtime_context.get_parallel', return_value=NS(tp_group=group)):
                mti.install()
                result = ModelRunner.sample(runner, output, NS(
                    is_prefill_only=False, seq_lens=torch.tensor([10]),
                    req_pool_indices=torch.tensor([0])))
            self.assertEqual(result.tolist(), [1])
            self.assertIs(output.next_token_logits, raw)
        finally:
            ModelRunner.sample = original
            if hasattr(ModelRunner, '_ds41_mti_installed'):
                del ModelRunner._ds41_mti_installed

    def test_entropy_is_before_temperature(self):
        self.assertAlmostEqual(float(entropy(torch.zeros(1, 4))), 1.38629436, places=6)
        self.assertLess(float(entropy(torch.tensor([[1000., 0., -1000.]]))), 1e-6)

    def test_cfg_matches_reference_log_probability_formula(self):
        torch.manual_seed(7)
        c, n = torch.randn(3, 100), torch.randn(3, 100)
        reference = 1.5 * c.log_softmax(-1) - .5 * n.log_softmax(-1)
        torch.testing.assert_close(guide(c, n, 1.5).softmax(-1), reference.softmax(-1))
        torch.testing.assert_close(guide(c, n, 1), c)

    def fake_runner(self, fail=False):
        table = torch.zeros(8, 64, dtype=torch.int32)
        table[1, :7] = torch.arange(8, 15)
        table[2, :8] = torch.arange(16, 24)
        # Row 1 has one free slot in its existing page, then a new page; row 2
        # begins a fresh page. Neither existing prefix page may be released.
        allocated = torch.tensor([15, 40, 48, 49], dtype=torch.int64)
        allocator = NS(page_size=8, alloc_extend=Mock(return_value=allocated), free=Mock())
        state = torch.arange(32.).view(8, 4)
        history = torch.arange(16).view(8, 2)
        backend = NS(forward_metadata=object(), tail_forward_metadata=object())
        runner = NS(token_to_kv_pool=NS(compress_state_pools=[NS(kv_score_buffer=NS(kv_score=state))],
                                       indexer_compress_state_pools=[]),
                    token_to_kv_pool_allocator=allocator,
                    req_to_token_pool=NS(req_to_token=table),
                    model=NS(model=NS(engram_hasher=NS(history=history))),
                    attn_backend=backend,
                    kv_index_translator=NS(own_plan=lambda fb: NS(bind=lambda *args: None)))
        calls = []
        def forward(fb):
            calls.append(fb)
            state.add_(100)
            history.add_(30)
            backend.forward_metadata = object()
            backend.tail_forward_metadata = object()
            if fail: raise RuntimeError('injected forward failure')
            return NS(logits_output=NS(next_token_logits=torch.ones(2, 10) * int(fb.input_ids[0])))
        runner.forward = forward
        batch = NS(req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
                   seq_lens=torch.tensor([7, 8], dtype=torch.int32),
                   seq_lens_cpu=torch.tensor([7, 8], dtype=torch.int32))
        return runner, batch, calls

    def test_branch_restores_state_and_frees_only_new_pages(self):
        runner, batch, calls = self.fake_runner()
        table = runner.req_to_token_pool.req_to_token.clone()
        state = runner.token_to_kv_pool.compress_state_pools[0].kv_score_buffer.kv_score.clone()
        hist = runner.model.model.engram_hasher.history.clone()
        meta = runner.attn_backend.forward_metadata
        result = negative_logits(runner, batch, torch.tensor([0, 1]), [10, 11])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].positions.tolist(), [7, 8])
        self.assertEqual(calls[1].positions.tolist(), [8, 9])
        torch.testing.assert_close(result, torch.full((2, 10), 11.))
        torch.testing.assert_close(runner.req_to_token_pool.req_to_token, table)
        torch.testing.assert_close(runner.token_to_kv_pool.compress_state_pools[0].kv_score_buffer.kv_score, state)
        torch.testing.assert_close(runner.model.model.engram_hasher.history, hist)
        self.assertIs(runner.attn_backend.forward_metadata, meta)
        self.assertEqual(runner.token_to_kv_pool_allocator.free.call_args.args[0].tolist(), [40, 48, 49])

    def test_failure_restores_state_and_releases_new_pages(self):
        runner, batch, calls = self.fake_runner(fail=True)
        table = runner.req_to_token_pool.req_to_token.clone()
        hist = runner.model.model.engram_hasher.history.clone()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            negative_logits(runner, batch, torch.tensor([0, 1]), [10, 11])
        torch.testing.assert_close(runner.req_to_token_pool.req_to_token, table)
        torch.testing.assert_close(runner.model.model.engram_hasher.history, hist)
        runner.token_to_kv_pool_allocator.free.assert_called_once()


if __name__ == '__main__':
    unittest.main()
