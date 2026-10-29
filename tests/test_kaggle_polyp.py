"""Focused checks for the data split and a complete training/prediction run."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from PIL import Image
import torch

import kaggle_polyp as pipeline


def make_far(root, positive_patients=6, normal_patients=3):
    for number in range(1, positive_patients + normal_patients + 1):
        patient = root / f"Patient{number}"
        for folder in ("Healthy", "Polyp", "BinaryMask"):
            (patient / folder).mkdir(parents=True)
        image = np.full((24, 32, 3), 40, dtype=np.uint8)
        Image.fromarray(image).save(patient / "Healthy" / "normal.jpg")
        if number <= positive_patients:
            mask = np.zeros((24, 32), dtype=np.uint8)
            mask[5:18, 9:22] = 255
            image[mask > 0] = 210
            Image.fromarray(image).save(patient / "Polyp" / "polyp.jpg")
            Image.fromarray(mask).save(patient / "BinaryMask" / "polyp.jpg")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_patient_split_is_disjoint_complete_and_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_far(root)
            splits = pipeline.discover_far_polyp(root, seed=42)
            repeated = pipeline.discover_far_polyp(root, seed=42)
            groups = {key: {row.group for row in rows} for key, rows in splits.items()}
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["val"] & groups["test"])
            self.assertEqual(len(set.union(*groups.values())), 9)
            self.assertEqual(sum(map(len, splits.values())), 15)
            self.assertEqual(splits, repeated)
            for rows in splits.values():
                self.assertTrue(any(row.label == "polyp" for row in rows))
            normal = next(
                row for rows in splits.values() for row in rows if row.label == "normal"
            )
            _, mask = pipeline.PolypDataset([normal], image_size=32)[0]
            self.assertEqual(float(mask.sum()), 0)

    def test_missing_positive_mask_and_too_few_patients_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_far(root, positive_patients=2, normal_patients=0)
            with self.assertRaisesRegex(ValueError, "3 positive patients"):
                pipeline.discover_far_polyp(root)
            (root / "Patient1/BinaryMask/polyp.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "mask names must match"):
                pipeline.discover_far_polyp(root)

    def test_unlabelled_missing_mask_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (32, 32)).save(image)
            sample = pipeline.Sample(image, None)
            with self.assertRaisesRegex(ValueError, "explicit normal label"):
                pipeline.PolypDataset([sample], 32)[0]

    def test_augmentation_keeps_masks_aligned_and_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[3:14, 11:28] = 255
            Image.fromarray(np.repeat(mask[:, :, None], 3, axis=2)).save(
                root / "image.png"
            )
            Image.fromarray(mask).save(root / "mask.png")
            data = pipeline.PolypDataset(
                [(root / "image.png", root / "mask.png")], 32, True
            )
            for epoch in range(8):
                data.set_epoch(epoch)
                image_tensor, mask_tensor = data[0]
                self.assertTrue(
                    torch.equal(image_tensor[0] > 0.5, mask_tensor[0].bool())
                )
                self.assertTrue(set(mask_tensor.unique().tolist()) <= {0.0, 1.0})
                self.assertEqual(image_tensor.dtype, torch.float32)
                repeated = data[0]
                self.assertTrue(torch.equal(image_tensor, repeated[0]))

    def test_metrics_separate_positive_overlap_and_normal_false_alarms(self):
        truth = torch.zeros(3, 1, 8, 8)
        truth[0, 0, 2:5, 2:5] = 1
        prediction = truth.clone()
        prediction[1, 0, 0, 0] = 1
        metrics, rows = pipeline.polyp_metrics(prediction, truth)
        self.assertEqual(metrics["positive_mean_dice"], 1)
        self.assertEqual(metrics["negative_false_positive_frame_rate"], 0.5)
        self.assertEqual(metrics["negative_mean_fp_area_fraction"], 1 / 128)
        self.assertEqual(rows[2]["dice"], 1)
        empty_metrics, _ = pipeline.polyp_metrics(torch.zeros_like(truth), truth)
        self.assertEqual(empty_metrics["positive_mean_dice"], 0)
        self.assertAlmostEqual(empty_metrics["dice"], 2 / 3)

    def test_downsampling_cannot_silently_relabel_a_polyp_as_normal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (128, 128)).save(root / "image.png")
            mask = Image.new("L", (128, 128))
            mask.putpixel((0, 0), 255)
            mask.save(root / "mask.png")
            data = pipeline.PolypDataset([(root / "image.png", root / "mask.png")], 16)
            with self.assertRaisesRegex(ValueError, "removes the entire polyp mask"):
                data[0]

    def test_no_normal_images_means_undefined_false_alarm_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Dataset"
            make_far(root, positive_patients=3, normal_patients=0)
            for path in root.glob("Patient*/Healthy/*.jpg"):
                path.unlink()
            output = Path(directory) / "run"
            config = dict(
                pipeline.CONFIG,
                data_root=str(root),
                output_dir=str(output),
                epochs=1,
                image_size=32,
                batch_size=2,
                base_channels=4,
                device="cpu",
                show_plots=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = pipeline.train(config)
            history = json.loads((output / "history.json").read_text())
            self.assertIsNone(history[0]["val_negative_fp_rate"])
            self.assertIsNone(result["negative_false_positive_frame_rate"])

    def test_training_manifest_checkpoint_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Dataset"
            make_far(root)
            output = Path(directory) / "run"
            config = dict(
                pipeline.CONFIG,
                data_root=str(root),
                output_dir=str(output),
                epochs=1,
                image_size=32,
                batch_size=2,
                base_channels=4,
                device="cpu",
                show_plots=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = pipeline.train(config)
                source = root / "Patient1/Polyp/polyp.jpg"
                prediction = pipeline.predict(
                    output / "best_model.pt",
                    source,
                    Path(directory) / "prediction",
                    "cpu",
                )
            self.assertTrue(np.isfinite(result["test_loss"]))
            self.assertGreater(result["positive_count"], 0)
            checkpoint = torch.load(
                output / "best_model.pt", weights_only=True, map_location="cpu"
            )
            self.assertEqual(checkpoint["preprocessing"]["image_size"], 32)
            with Image.open(prediction["mask"]) as mask:
                self.assertEqual(mask.size, (32, 24))
                self.assertTrue(set(np.unique(np.asarray(mask))) <= {0, 255})
            splits, _ = pipeline.load_manifest(root, output / "split.csv")
            self.assertEqual(sum(map(len, splits.values())), 15)
            with (output / "split.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["group"] = "same-patient"
            bad = Path(directory) / "bad.csv"
            with bad.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "groups cross splits"):
                pipeline.load_manifest(root, bad)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    pipeline.train(config)


if __name__ == "__main__":
    unittest.main()
