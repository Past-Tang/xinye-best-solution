import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
from checkpoint_utils import load_task_checkpoint
import tools_fusion_to_csv as fusion
import tools_activity_tcn_train as activity


class InferenceContractTests(unittest.TestCase):
    def test_zero_topk_selects_nothing(self):
        pred = fusion.refrate_topk(np.ones((1, 5)))
        self.assertEqual(pred.tolist(), [[1, 1, 0, 0, 0]])

    def test_topk_counts_match_rates(self):
        x = np.random.default_rng(5).random((1000, 5))
        self.assertEqual(fusion.refrate_topk(x).sum(0).tolist(), [987, 780, 150, 104, 450])

    def test_misaligned_and_invalid_inputs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'probs.npy'; i = root / 'ids.txt'; j = root / 'other.txt'
            np.save(p, np.zeros((2, 5)))
            i.write_text('0001\n0002\n'); j.write_text('0002\n0001\n')
            with self.assertRaisesRegex(ValueError, 'align'):
                fusion.load_test([p, p], [i, j])
            i.write_text('0001\n0001\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                fusion.load_test([p], [i])
            i.write_text('0001\n0002\n'); np.save(p, np.full((2, 5), np.nan))
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                fusion.load_test([p], [i])

    def test_slim_load_allows_only_declared_frozen_base(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        for p in model[0].parameters(): p.requires_grad_(False)
        head = {k: torch.ones_like(v) for k, v in model.state_dict().items() if k.startswith('1.')}
        load_task_checkpoint(model, {'slim_keep': head, 'dropped_from_base': ['0.weight', '0.bias']})
        self.assertTrue(torch.equal(model[1].weight, torch.ones_like(model[1].weight)))
        model.add_module('audio_merge', torch.nn.Linear(1, 1))
        with self.assertRaisesRegex(ValueError, 'audio_merge'):
            load_task_checkpoint(model, {'slim_keep': head, 'dropped_from_base': ['0.weight', '0.bias']})

    def test_compiled_ema_only_checkpoint(self):
        model = torch.nn.Linear(2, 1)
        state = {'_orig_mod.' + k: torch.ones_like(v) for k, v in model.state_dict().items()}
        load_task_checkpoint(model, {'model_ema': state})
        self.assertEqual(model.bias.item(), 1)

    def test_activity_import_resolves_feature_helpers(self):
        self.assertEqual(activity.OUT_NAMES, ['c', 'na', 'i', 'bc', 't'])
        self.assertTrue(callable(activity.load_audio_arrays))


if __name__ == '__main__':
    unittest.main()
