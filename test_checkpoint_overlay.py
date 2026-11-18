"""CPU-only checks for tensor replacement and model-switch recovery."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import checkpoint_overlay as overlay

loader = importlib.machinery.SourceFileLoader('modelctl_tests', str(Path(__file__).with_name('modelctl')))
spec = importlib.util.spec_from_loader(loader.name, loader)
modelctl = importlib.util.module_from_spec(spec)
loader.exec_module(modelctl)


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.patch = Path(self.temporary.name) / 'patch.safetensors'
        self.values = {name: torch.full((2, 2), 1.5).to(torch.float8_e4m3fn)
                       for name in overlay.EXPECTED_KEYS}
        save_file(self.values, self.patch)

    def test_changes_only_selected_tensors_and_keeps_order(self):
        names = sorted(overlay.EXPECTED_KEYS)
        untouched = torch.tensor([17])
        original = [(name, torch.zeros((2, 2)).to(torch.float8_e4m3fn)) for name in names]
        original.insert(9, ('layers.2.ffn.experts.0.w1.weight', untouched))
        seen = set()
        result = list(overlay.replacement_weights(original, self.patch, seen))
        self.assertEqual([name for name, _ in original], [name for name, _ in result])
        self.assertIs(result[9][1], untouched)
        self.assertEqual(seen, overlay.EXPECTED_KEYS)
        for name, value in result:
            if name in self.values:
                self.assertTrue(torch.equal(value.view(torch.uint8), self.values[name].view(torch.uint8)))
        self.assertTrue(all(torch.count_nonzero(value.float()) == 0
                            for name, value in original if name in overlay.EXPECTED_KEYS))

    def test_rejects_shape_mismatch(self):
        name = next(iter(overlay.EXPECTED_KEYS))
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            list(overlay.replacement_weights([(name, torch.zeros(3))], self.patch, set()))

    def test_rejects_duplicate_replacement(self):
        name = next(iter(overlay.EXPECTED_KEYS))
        value = self.values[name]
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            list(overlay.replacement_weights([(name, value), (name, value)], self.patch, set()))


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / '.runs').mkdir()
        self.original = b'{\n  "model_path": "original-shards",\n  "context_length": 1048576,\n  "served_model_name": "existing-alias"\n}\n'
        (self.root / 'serving.json').write_bytes(self.original)
        self.validate = patch.object(modelctl, 'validate_patch', return_value={})
        self.validate.start()
        self.addCleanup(self.validate.stop)

    def test_startup_failure_restores_exact_original_configuration(self):
        actions = []
        def control(root, action):
            actions.append(action)
            if actions == ['stop', 'start']:
                raise subprocess.CalledProcessError(1, ['start'])
        with patch.object(modelctl, 'controller', control), patch.object(modelctl, 'validate_loaded'):
            with self.assertRaisesRegex(RuntimeError, 'previous service restored'):
                modelctl.switch(self.root, 'heretic')
        self.assertEqual(actions, ['stop', 'start', 'stop', 'start'])
        self.assertEqual((self.root / 'serving.json').read_bytes(), self.original)

    def test_failed_overlay_audit_also_rolls_back(self):
        with patch.object(modelctl, 'controller') as controller, patch.object(
                modelctl, 'validate_loaded', side_effect=[RuntimeError('Missing rank'), None]):
            with self.assertRaisesRegex(RuntimeError, 'previous service restored'):
                modelctl.switch(self.root, 'heretic')
        self.assertEqual([call.args[1] for call in controller.call_args_list], ['stop', 'start', 'stop', 'start'])
        self.assertEqual((self.root / 'serving.json').read_bytes(), self.original)

    def test_interrupt_during_stop_restores_service(self):
        actions = []
        def control(root, action):
            actions.append(action)
            if len(actions) == 1:
                raise KeyboardInterrupt('stop interrupted')
        with patch.object(modelctl, 'controller', control), patch.object(modelctl, 'validate_loaded'):
            with self.assertRaisesRegex(RuntimeError, 'previous service restored'):
                modelctl.switch(self.root, 'heretic')
        self.assertEqual(actions, ['stop', 'stop', 'start'])
        self.assertEqual((self.root / 'serving.json').read_bytes(), self.original)

    def test_invalid_patch_does_not_stop_service(self):
        with patch.object(modelctl, 'validate_patch', side_effect=ValueError('bad checksum')), patch.object(
                modelctl, 'controller') as controller:
            with self.assertRaisesRegex(ValueError, 'bad checksum'):
                modelctl.switch(self.root, 'heretic')
            controller.assert_not_called()
        self.assertEqual((self.root / 'serving.json').read_bytes(), self.original)

    def test_success_changes_only_patch_selection(self):
        with patch.object(modelctl, 'controller'), patch.object(modelctl, 'validate_loaded'):
            modelctl.switch(self.root, 'heretic')
        current = json.loads((self.root / 'serving.json').read_text())
        self.assertEqual(current.pop('checkpoint_patch'), str(overlay.PATCH))
        self.assertEqual(current, json.loads(self.original))


if __name__ == '__main__':
    unittest.main()
