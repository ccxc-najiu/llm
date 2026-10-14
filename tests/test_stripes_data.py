"""Contract and corruption checks for the dataset, without training a model."""

import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stripes", ROOT / "scripts/generate_stripes.py")
stripes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stripes)


class StripeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="stripes-data-tests-")
        cls.root = Path(cls.temp.name).resolve()
        cls.config = stripes.read_json(ROOT / "configs/stripes-v1.json")
        for split in cls.config["splits"].values():
            split["per_cell_class"] = 2
        cls.output = cls.root / "original"
        cls.report = stripes.generate(cls.config, cls.output)

    @classmethod
    def tearDownClass(cls):
        # Only the unique directory created by TemporaryDirectory is removed.
        cls.temp.cleanup()

    def test_label_axes_with_hand_computed_pattern(self):
        params = {"width": 2, "phase": 1, "brightness": 0.125, "contrast": 0.5,
                  "noise_amplitude": 0, "noise_seed": 1, "label": 0}
        horizontal = stripes.render_sample(params, 4)[0]
        expected = np.array([[0.125] * 4, [0.625] * 4, [0.625] * 4, [0.125] * 4], dtype=np.float32)
        np.testing.assert_array_equal(horizontal, expected)
        params["label"] = 1
        np.testing.assert_array_equal(stripes.render_sample(params, 4)[0], expected.T)

    def test_reproducibility_and_existing_output_protection(self):
        second = self.root / "reproduced"
        stripes.generate(copy.deepcopy(self.config), second)
        self.assertEqual(stripes.read_json(self.output / "checksums.json"),
                         stripes.read_json(second / "checksums.json"))
        self.assertEqual(stripes.verify(second), self.report)
        before = stripes.sha_file(second / "train.npz")
        with self.assertRaises(FileExistsError):
            stripes.generate(self.config, second)
        self.assertEqual(stripes.sha_file(second / "train.npz"), before)

    def test_modified_file_rejected_by_checksums(self):
        damaged = self.root / "bad-checksum"
        shutil.copytree(self.output, damaged)
        with (damaged / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "File checksum mismatch"):
            stripes.verify(damaged)

    def test_cross_split_duplicate_rejected_even_with_valid_pixel_hash(self):
        damaged = self.root / "bad-duplicate"
        shutil.copytree(self.output, damaged)
        source_rows = stripes.read_rows(damaged / "train.jsonl")
        target_rows = stripes.read_rows(damaged / "validation.jsonl")
        with np.load(damaged / "train.npz", allow_pickle=False) as data:
            source_image = data["images"][0].copy()
        with np.load(damaged / "validation.npz", allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        target_rows[0] = {**source_rows[0], "id": target_rows[0]["id"], "split": "validation"}
        arrays["images"][0] = source_image
        arrays["labels"][0] = source_rows[0]["label"]
        np.savez_compressed(damaged / "validation.npz", **arrays)
        stripes.write_rows(damaged / "validation.jsonl", target_rows)
        with self.assertRaisesRegex(ValueError, "Repeated image"):
            stripes.verify(damaged, check_hashes=False)

    def test_missing_diagnostic_relation_rejected(self):
        damaged = self.root / "bad-relations"
        shutil.copytree(self.output, damaged)
        rows = stripes.read_rows(damaged / "diagnostic-relations.jsonl")
        stripes.write_rows(damaged / "diagnostic-relations.jsonl", rows[1:])
        with self.assertRaisesRegex(ValueError, "Diagnostic relation coverage"):
            stripes.verify(damaged, check_hashes=False)

    def test_heldout_configuration_cannot_leak(self):
        invalid = copy.deepcopy(self.config)
        invalid["splits"]["train"]["heldout_only"] = True
        with self.assertRaisesRegex(ValueError, "Held-out combination"):
            stripes.validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
