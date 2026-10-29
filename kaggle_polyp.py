"""Small binary polyp segmentation baseline for FAR-POLYP-SEG.
Set an explicit data_root in CONFIG or pass --data-root. Training uses patient
splits. This is an educational baseline, not a medical tool or a clinical claim.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import platform
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Edit these values to run the whole file in a Kaggle notebook cell.
CONFIG: dict[str, Any] = {
    "data_root": "",  # EDIT: explicit dataset directory under /kaggle/input
    "image_dir": "images",  # EDIT: used only for a generic pair demo
    "mask_dir": "masks",  # EDIT: used only for a generic pair demo
    "output_dir": "/kaggle/working/polyp_baseline_run",  # EDIT: use a new run folder
    "inference_image": None,  # EDIT: optional image to predict after training
    "manifest_csv": None,  # EDIT: optional image,mask,split,group,label CSV
    "image_size": 256,
    "epochs": 5,
    "batch_size": 8,
    "base_channels": 16,
    "seed": 42,
    "train_fraction": 0.8,
    "val_fraction": 0.1,
    "test_fraction": 0.1,
    "device": "auto",
    "show_plots": True,
    "balanced_sampling": True,
    "allow_random_split": False,  # Set only for synthetic/generic demos.
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
MASK_THRESHOLD = 127
PREDICTION_THRESHOLD = 0.5


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None
    group: str | None = None
    label: str | None = None  # "polyp" or verified "normal"


def _parts(
    item: tuple[Path, Path] | Sample,
) -> tuple[Path, Path | None, str | None, str | None]:
    if isinstance(item, Sample):
        return item.image, item.mask, item.group, item.label
    return item[0], item[1], None, None


def _mask_array(mask: Image.Image) -> np.ndarray:
    values = np.asarray(mask.convert("L"), dtype=np.uint8)
    threshold = 0 if values.size and int(values.max()) <= 1 else MASK_THRESHOLD
    return (values > threshold).astype(np.float32)


def _binary_mask(mask: Image.Image) -> Image.Image:
    return Image.fromarray((_mask_array(mask) * 255).astype(np.uint8))


def _root(data_root: str | Path) -> Path:
    if not str(data_root).strip():
        raise ValueError(
            "Set CONFIG['data_root'] or pass --data-root. No dataset folder is auto-selected."
        )
    path = Path(data_root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist or is not a directory: {path}"
        )
    return path


def _inside(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative to data_root: {value}")
    result = (root.resolve() / path).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside data_root: {value}") from exc
    return result


def _relative(path: Path | None, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix() if path else ""


def _validate(samples: Sequence[tuple[Path, Path] | Sample]) -> None:
    if not samples:
        raise ValueError("No image/mask samples were found.")
    for item in samples:
        image_path, mask_path, _, label = _parts(item)
        if mask_path is None and label != "normal":
            raise ValueError(
                f"A missing mask requires an explicit normal label: {image_path}"
            )
        try:
            with Image.open(image_path) as image:
                image_size = image.size
                image.verify()
            if mask_path is None:
                continue
            with Image.open(mask_path) as mask:
                mask_size, foreground = mask.size, bool(_mask_array(mask).any())
        except Exception as exc:
            raise ValueError(
                f"Could not read sample {image_path}, {mask_path}: {exc}"
            ) from exc
        if image_size != mask_size:
            raise ValueError(
                f"Image/mask dimensions differ for {image_path.name}: {image_size} vs {mask_size}"
            )
        if label == "polyp" and not foreground:
            raise ValueError(f"A polyp-labeled row needs a nonempty mask: {image_path}")
        if label == "normal" and foreground:
            raise ValueError(f"A normal-labeled row has a nonempty mask: {image_path}")


def discover_pairs(
    data_root: str | Path,
    image_dir: str | Path = "images",
    mask_dir: str | Path = "masks",
) -> list[tuple[Path, Path]]:
    """Pair nonrecursive image and mask files by exact filename stem."""
    root = _root(data_root)
    folders = (
        _inside(root, image_dir, "image_dir"),
        _inside(root, mask_dir, "mask_dir"),
    )
    indexes = []
    for folder, label in zip(folders, ("image", "mask")):
        if not folder.is_dir():
            raise FileNotFoundError(
                f"{label.title()} directory does not exist: {folder}"
            )
        index: dict[str, Path] = {}
        for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if path.stem in index:
                    raise ValueError(
                        f"Duplicate {label} stem {path.stem!r} in {folder}"
                    )
                index[path.stem] = path.resolve()
        indexes.append(index)
    images, masks = indexes
    if not images or not masks:
        raise ValueError(
            f"No pairs found: {len(images)} images and {len(masks)} masks."
        )
    missing, extra = (
        sorted(images.keys() - masks.keys()),
        sorted(masks.keys() - images.keys()),
    )
    if missing or extra:
        raise ValueError(
            f"Unmatched stems. Images without masks={missing[:10]}, masks without images={extra[:10]}"
        )
    return [(images[key], masks[key]) for key in sorted(images)]


def _counts(n: int, fractions: Sequence[float], require_all: bool = True) -> list[int]:
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("Train, validation, and test fractions must be positive.")
    if require_all and n < 3:
        raise ValueError(f"At least 3 items are needed for nonempty splits, found {n}.")
    raw = [n * value / sum(fractions) for value in fractions]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[
        : n - sum(counts)
    ]:
        counts[index] += 1
    if require_all:
        for index in range(3):
            if counts[index] == 0:
                donor = max(range(3), key=lambda i: counts[i])
                counts[donor] -= 1
                counts[index] += 1
    return counts


def make_splits(
    pairs: Sequence[tuple[Path, Path]],
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[tuple[Path, Path]]]:
    """Create deterministic image-level splits for generic demo data only."""
    _validate(pairs)
    ordered = sorted(pairs, key=lambda pair: (pair[0].as_posix(), pair[1].as_posix()))
    random.Random(seed).shuffle(ordered)
    counts = _counts(len(ordered), (train_fraction, val_fraction, test_fraction))
    a, b = counts[0], counts[0] + counts[1]
    return {"train": ordered[:a], "val": ordered[a:b], "test": ordered[b:]}


def load_manifest(
    data_root: str | Path, manifest_csv: str | Path
) -> tuple[dict[str, list[Sample]], dict[str, str]]:
    """Load an explicit patient-grouped CSV with image,mask,split,group,label columns."""
    root = _root(data_root)
    manifest = Path(manifest_csv).expanduser()
    if not manifest.is_absolute():
        manifest = root / manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest}")
    splits: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    groups: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    seen_masks: set[str] = set()
    labels = {
        "polyp": "polyp",
        "positive": "polyp",
        "1": "polyp",
        "label1": "polyp",
        "normal": "normal",
        "negative": "normal",
        "0": "normal",
        "label0": "normal",
    }
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = {
            name.strip().lower(): name for name in (reader.fieldnames or []) if name
        }
        missing = {"image", "mask", "split", "group", "label"} - columns.keys()
        if missing:
            raise ValueError(f"Manifest CSV is missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            image_value = (row.get(columns["image"]) or "").strip()
            mask_value = (row.get(columns["mask"]) or "").strip()
            split = (row.get(columns["split"]) or "").strip().lower()
            group = (row.get(columns["group"]) or "").strip()
            label_value = (row.get(columns["label"]) or "").strip().lower()
            label = labels.get(label_value)
            if not image_value or not group or not label:
                raise ValueError(
                    f"Manifest row {line} needs image, group, and label=polyp/normal."
                )
            if split not in SPLITS:
                raise ValueError(
                    f"Manifest row {line} split must be train, val, or test."
                )
            image = _inside(root, image_value, f"image on row {line}")
            mask = (
                _inside(root, mask_value, f"mask on row {line}") if mask_value else None
            )
            if not image.is_file() or (mask and not mask.is_file()):
                raise FileNotFoundError(
                    f"Missing manifest file on row {line}: {image}, {mask}"
                )
            if label == "polyp" and mask is None:
                raise ValueError(f"Polyp row {line} must name its binary mask.")
            image_key, mask_key = _relative(image, root), _relative(mask, root)
            if image_key in image_splits or (mask_key and mask_key in seen_masks):
                raise ValueError(f"Duplicate image or mask in manifest on row {line}.")
            image_splits[image_key], groups[image_key] = split, group
            if mask_key:
                seen_masks.add(mask_key)
            splits[split].append(Sample(image, mask, group, label))
    group_splits: dict[str, set[str]] = {}
    for sample in (sample for rows in splits.values() for sample in rows):
        group_splits.setdefault(sample.group or "", set()).add(
            image_splits[_relative(sample.image, root)]
        )
    leaking = [group for group, names in group_splits.items() if len(names) != 1]
    if leaking:
        raise ValueError(f"Patient groups cross splits: {leaking[:10]}")
    for name in SPLITS:
        if not splits[name]:
            raise ValueError(f"Manifest has no {name} rows.")
        splits[name].sort(key=lambda sample: sample.image.as_posix())
        _validate(splits[name])
    return splits, groups


def discover_far_polyp(
    data_root: str | Path,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[Sample]] | None:
    """Read PatientN/Healthy, Polyp, and BinaryMask folders and split by patient."""
    root = _root(data_root)
    patients = sorted(
        (
            p
            for p in root.iterdir()
            if p.is_dir() and re.fullmatch(r"Patient\d+", p.name)
        ),
        key=lambda p: p.name,
    )
    if not patients:
        return None
    patient_samples: dict[str, list[Sample]] = {}
    positive_patients: set[str] = set()

    def jpgs(folder: Path) -> dict[str, Path]:
        if not folder.is_dir():
            return {}
        return {
            p.name: p.resolve()
            for p in sorted(folder.iterdir(), key=lambda x: x.name.casefold())
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".jpg"
        }

    for patient in patients:
        healthy, polyps, masks = (
            jpgs(patient / name) for name in ("Healthy", "Polyp", "BinaryMask")
        )
        missing, extra = (
            sorted(polyps.keys() - masks.keys()),
            sorted(masks.keys() - polyps.keys()),
        )
        if missing or extra:
            raise ValueError(
                f"{patient.name} mask names must match Polyp images; missing={missing[:5]}, extra={extra[:5]}"
            )
        rows = [Sample(path, None, patient.name, "normal") for path in healthy.values()]
        rows += [
            Sample(polyps[name], masks[name], patient.name, "polyp")
            for name in sorted(polyps)
        ]
        if rows:
            patient_samples[patient.name] = rows
        if polyps:
            positive_patients.add(patient.name)
    if not patient_samples:
        raise ValueError(f"No Healthy or Polyp JPG frames found under {root}.")
    _validate([sample for rows in patient_samples.values() for sample in rows])
    if len(positive_patients) < 3:
        raise ValueError(
            "At least 3 positive patients are needed for patient-disjoint train/val/test splits."
        )
    fractions = (train_fraction, val_fraction, test_fraction)
    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for ids, required in (
        (sorted(positive_patients), True),
        (sorted(set(patient_samples) - positive_patients), False),
    ):
        rng.shuffle(ids)
        counts = _counts(len(ids), fractions, require_all=required)
        start = 0
        for split, count in zip(SPLITS, counts):
            for patient_id in ids[start : start + count]:
                assignments[patient_id] = split
            start += count
    splits = {name: [] for name in SPLITS}
    for patient_id, rows in patient_samples.items():
        splits[assignments[patient_id]].extend(rows)
    for name in SPLITS:
        splits[name].sort(key=lambda sample: sample.image.as_posix())
        if not any(sample.label == "polyp" for sample in splits[name]):
            raise ValueError(f"{name} split has no positive patient.")
    return splits


class PolypDataset(Dataset):
    """RGB image and binary mask tensors, with synchronized train augmentation."""

    def __init__(
        self,
        samples: Sequence[tuple[Path, Path] | Sample],
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42,
    ):
        if image_size < 16 or not samples:
            raise ValueError(
                "image_size must be at least 16 and samples cannot be empty."
            )
        self.samples, self.size, self.augment, self.seed, self.epoch = (
            list(samples),
            image_size,
            augment,
            seed,
            0,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _images(self, index: int) -> tuple[Image.Image, Image.Image]:
        image_path, mask_path, _, label = _parts(self.samples[index])
        with Image.open(image_path) as file:
            image = file.convert("RGB")
        if mask_path is None:
            if label != "normal":
                raise ValueError(
                    f"Missing mask without explicit normal label: {image_path}"
                )
            mask = Image.new("L", image.size, 0)
        else:
            with Image.open(mask_path) as file:
                mask = file.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask dimensions differ for {image_path.name}.")
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        native_mask = _binary_mask(mask)
        mask = native_mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if native_mask.getbbox() is not None and mask.getbbox() is None:
            raise ValueError(
                f"image_size={self.size} removes the entire polyp mask in {image_path}. "
                "Use a larger image_size."
            )
        return image, mask

    def _augment(
        self, image: Image.Image, mask: Image.Image, index: int
    ) -> tuple[Image.Image, Image.Image]:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        transpose = Image.Transpose
        if rng.random() < 0.5:
            image, mask = (
                image.transpose(transpose.FLIP_LEFT_RIGHT),
                mask.transpose(transpose.FLIP_LEFT_RIGHT),
            )
        if rng.random() < 0.5:
            image, mask = (
                image.transpose(transpose.FLIP_TOP_BOTTOM),
                mask.transpose(transpose.FLIP_TOP_BOTTOM),
            )
        rotations = (
            None,
            transpose.ROTATE_90,
            transpose.ROTATE_180,
            transpose.ROTATE_270,
        )
        turn = rotations[rng.randrange(4)]
        if turn is not None:
            image, mask = image.transpose(turn), mask.transpose(turn)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._images(index)
        if self.augment:
            image, mask = self._augment(image, mask, index)
        x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        y = _mask_array(mask)[None].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = next((n for n in (8, 4, 2, 1) if out_channels % n == 0), 1)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.ReLU(True),
    )


class TinyUNet(nn.Module):
    """Small four-level U-Net with GroupNorm and one foreground logit channel."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4.")
        b = base_channels
        self.e1, self.e2, self.e3, self.e4 = (
            _block(in_channels, b),
            _block(b, 2 * b),
            _block(2 * b, 4 * b),
            _block(4 * b, 8 * b),
        )
        self.mid = _block(8 * b, 16 * b)
        self.d4, self.d3, self.d2, self.d1 = (
            _block(24 * b, 8 * b),
            _block(12 * b, 4 * b),
            _block(6 * b, 2 * b),
            _block(3 * b, b),
        )
        self.pool, self.out = nn.MaxPool2d(2), nn.Conv2d(b, 1, 1)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        x = self.mid(self.pool(d))
        x = self.d4(torch.cat((self._up(x, d), d), 1))
        x = self.d3(torch.cat((self._up(x, c), c), 1))
        x = self.d2(torch.cat((self._up(x, b), b), 1))
        x = self.d1(torch.cat((self._up(x, a), a), 1))
        return self.out(x)


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    dice = (2 * (probs * targets).sum(1) + 1e-6) / (
        probs.sum(1) + targets.sum(1) + 1e-6
    )
    return (
        F.binary_cross_entropy_with_logits(logits, targets.reshape_as(logits))
        + 1
        - dice.mean()
    )


def overlap_scores(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Mean per-image Dice and IoU. Empty prediction against empty target scores 1."""
    metrics, _ = polyp_metrics(predictions, targets, threshold)
    return {"dice": metrics["dice"], "iou": metrics["iou"]}


def polyp_metrics(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = PREDICTION_THRESHOLD,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Report positive overlap and negative false positives per frame at threshold 0.5."""
    p, y = (
        torch.as_tensor(predictions).detach().float(),
        torch.as_tensor(targets).detach().float(),
    )
    if p.ndim == 2:
        p, y = p[None, None], y[None, None]
    elif p.ndim == 3:
        p, y = p[:, None], y[:, None]
    if p.shape != y.shape or p.ndim != 4:
        raise ValueError("Expected matching NCHW predictions and targets.")
    p, y = p >= threshold, y >= 0.5
    rows, pixels = [], y[0].numel()
    for pred, truth in zip(p, y):
        pred, truth = pred.flatten(), truth.flatten()
        inter, pred_n, true_n = (
            int((pred & truth).sum()),
            int(pred.sum()),
            int(truth.sum()),
        )
        union = pred_n + true_n - inter
        dice = 1.0 if pred_n + true_n == 0 else 2 * inter / (pred_n + true_n)
        iou = 1.0 if union == 0 else inter / union
        positive = true_n > 0
        rows.append(
            {
                "label": "polyp" if positive else "normal",
                "dice": dice,
                "iou": iou,
                "predicted_positive_pixels": pred_n,
                "predicted_positive_fraction": pred_n / pixels,
                "frame_prediction": bool(pred_n),
                "frame_threshold": threshold,
            }
        )
    return _summarize_rows(rows), rows


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["label"] == "polyp"]
    negative = [row for row in rows if row["label"] == "normal"]
    detected = sum(bool(row["frame_prediction"]) for row in positive)
    false_positives = [row for row in negative if row["frame_prediction"]]
    return {
        "dice": float(np.mean([row["dice"] for row in rows])),
        "iou": float(np.mean([row["iou"] for row in rows])),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "positive_mean_dice": float(np.mean([row["dice"] for row in positive]))
        if positive
        else None,
        "positive_mean_iou": float(np.mean([row["iou"] for row in positive]))
        if positive
        else None,
        "positive_frame_sensitivity": detected / len(positive) if positive else None,
        "negative_false_positive_frame_rate": len(false_positives) / len(negative)
        if negative
        else None,
        "negative_mean_fp_area_fraction": float(
            np.mean([row["predicted_positive_fraction"] for row in negative])
        )
        if negative
        else None,
        "negative_mean_fp_pixels": float(
            np.mean([row["predicted_positive_pixels"] for row in negative])
        )
        if negative
        else None,
    }


def _preview_indices(
    items: Sequence[tuple[Path, Path] | Sample], maximum: int = 4
) -> list[int]:
    positives = [i for i, item in enumerate(items) if _sample_label(item)]
    negatives = [i for i, item in enumerate(items) if not _sample_label(item)]
    chosen = positives[: maximum // 2] + negatives[: maximum // 2]
    chosen += [i for i in range(len(items)) if i not in chosen][: maximum - len(chosen)]
    return chosen


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available.")
    if value not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda.")
    return torch.device(value)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _sample_label(item: tuple[Path, Path] | Sample) -> int:
    image, mask, _, label = _parts(item)
    if label:
        return int(label == "polyp")
    if mask is None:
        raise ValueError(f"Cannot infer label without mask or explicit label: {image}")
    with Image.open(mask) as file:
        return int(_mask_array(file).any())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_splits(
    path: Path, root: Path, splits: dict[str, list[tuple[Path, Path] | Sample]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("image", "mask", "split", "group", "label")
        )
        writer.writeheader()
        for split, items in splits.items():
            for item in items:
                image, mask, group, label = _parts(item)
                writer.writerow(
                    {
                        "image": _relative(image, root),
                        "mask": _relative(mask, root),
                        "split": split,
                        "group": group or "",
                        "label": label
                        or ("polyp" if _sample_label(item) else "normal"),
                    }
                )


def _prepare_output(path: str | Path, allow_existing: bool = False) -> Path:
    result = Path(path).expanduser().resolve()
    if result.exists() and not result.is_dir():
        raise FileExistsError(f"Output path is not a directory: {result}")
    if result.exists() and not allow_existing and any(result.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {result}. Choose a new run folder."
        )
    result.mkdir(parents=True, exist_ok=True)
    return result


def _plot_grid(
    samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    path: Path,
    title: str,
    show: bool,
) -> None:
    if not samples:
        return
    fig, axes = plt.subplots(
        len(samples), 2, figsize=(8, 3 * len(samples)), squeeze=False
    )
    for i, (image, mask) in enumerate(samples):
        axes[i, 0].imshow(image.permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Mask")
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_history(
    history: list[dict[str, float | None]], path: Path, show: bool
) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="positive Dice")
    axes[1].plot(epochs, [row["val_iou"] for row in history], label="positive IoU")
    axes[1].set_title("Validation overlap")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_test(
    items: Sequence[tuple[Path, Path] | Sample],
    previews: Sequence[tuple[int, torch.Tensor]],
    size: int,
    path: Path,
    show: bool,
    maximum: int = 4,
) -> None:
    count = min(maximum, len(previews))
    if not count:
        return
    fig, axes = plt.subplots(count, 3, figsize=(11, 3 * count), squeeze=False)
    for i, (item_index, probabilities) in enumerate(previews[:count]):
        item = items[item_index]
        image_path, mask_path, _, label = _parts(item)
        with Image.open(image_path) as file:
            image = file.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        if mask_path:
            with Image.open(mask_path) as file:
                mask = _binary_mask(file).resize((size, size), Image.Resampling.NEAREST)
        else:
            mask = Image.new("L", (size, size), 0)
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(_mask_array(mask), cmap="gray")
        axes[i, 1].set_title("True mask")
        axes[i, 2].imshow(probabilities[0].numpy() >= PREDICTION_THRESHOLD, cmap="gray")
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle("Test-set predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_previews: bool = False,
):
    model.eval()
    losses = []
    rows = []
    previews = {}
    wanted = _preview_indices(loader.dataset.samples) if include_previews else []
    offset = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(float(segmentation_loss(logits, y).item()))
        probs = torch.sigmoid(logits).cpu()
        _, batch_rows = polyp_metrics(probs, y.cpu())
        rows.extend(batch_rows)
        for local, global_index in enumerate(range(offset, offset + len(probs))):
            if global_index in wanted:
                previews[global_index] = probs[local]
        offset += len(probs)
    return (
        float(np.mean(losses)),
        _summarize_rows(rows),
        [(i, previews[i]) for i in wanted],
        rows,
    )


def _save_checkpoint(
    path: Path,
    model: TinyUNet,
    image_size: int,
    base: int,
    seed: int,
    epoch: int,
    dice: float,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model": {"in_channels": 3, "base_channels": base},
            "preprocessing": {
                "image_size": image_size,
                "image_mode": "RGB",
                "image_resampling": "BILINEAR",
                "mask_resampling": "NEAREST",
                "mask_threshold": MASK_THRESHOLD,
            },
            "threshold": PREDICTION_THRESHOLD,
            "seed": seed,
            "best_epoch": epoch,
            "best_val_positive_dice": dice,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        path,
    )


def _save_history(history: list[dict[str, float | None]], output: Path) -> None:
    _write_json(output / "history.json", history)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _save_test_rows(
    path: Path,
    root: Path,
    items: Sequence[tuple[Path, Path] | Sample],
    rows: Sequence[dict[str, Any]],
) -> None:
    fields = (
        "image",
        "group",
        "label",
        "dice",
        "iou",
        "predicted_positive_pixels",
        "predicted_positive_fraction",
        "frame_prediction",
        "frame_threshold",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item, row in zip(items, rows):
            image, _, group, _ = _parts(item)
            writer.writerow(
                {"image": _relative(image, root), "group": group or "", **row}
            )


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Train, select by positive validation Dice, reload best weights, test once."""
    root = _root(config.get("data_root", ""))
    size, epochs, batch, base, seed = (
        int(config.get(k, default))
        for k, default in (
            ("image_size", 256),
            ("epochs", 5),
            ("batch_size", 8),
            ("base_channels", 16),
            ("seed", 42),
        )
    )
    if size < 16 or epochs < 1 or batch < 1:
        raise ValueError("Use image_size>=16, epochs>=1, and batch_size>=1.")
    fractions = (
        float(config.get("train_fraction", 0.8)),
        float(config.get("val_fraction", 0.1)),
        float(config.get("test_fraction", 0.1)),
    )
    _seed(seed)
    device = _device(str(config.get("device", "auto")))
    manifest = config.get("manifest_csv")
    far_folders = False
    if manifest:
        splits, _ = load_manifest(root, manifest)
    else:
        splits = discover_far_polyp(root, seed, *fractions)
        far_folders = splits is not None
        if splits is None:
            if not config.get("allow_random_split", False):
                raise ValueError(
                    "No PatientN FAR folders or manifest found. Use --allow-random-split only for demos."
                )
            print(
                "WARNING: random image split enabled. Do not use it for patient video data."
            )
            splits = make_splits(
                discover_pairs(
                    root,
                    config.get("image_dir", "images"),
                    config.get("mask_dir", "masks"),
                ),
                seed,
                *fractions,
            )
    for split in SPLITS:
        if not any(_sample_label(item) for item in splits[split]):
            raise ValueError(f"{split} split needs at least one positive polyp frame.")
    output = _prepare_output(config.get("output_dir", ""))
    show = bool(config.get("show_plots", True))
    config_file = dict(
        config,
        resolved_data_root=str(root),
        resolved_output_dir=str(output),
        device_used=str(device),
        versions=_versions(),
        split_counts={s: len(splits[s]) for s in SPLITS},
        dataset="FAR-POLYP-SEG",
        split_policy="patient-grouped manifest"
        if manifest
        else (
            "patient-grouped FAR folders"
            if far_folders
            else "explicit random demo split"
        ),
        metric_image_size=size,
        prediction_threshold=PREDICTION_THRESHOLD,
        negative_frame_rule="Any predicted pixel counts as FP; no 100-pixel component gate.",
    )
    _write_json(output / "config.json", config_file)
    _write_splits(output / "split.csv", root, splits)
    print(f"Using device: {device}; threshold: {PREDICTION_THRESHOLD}")
    print(
        f"Metrics use resized {size}x{size} masks. Negative frame FPR uses any pixel, without the authors' 100-pixel component gate."
    )
    for split in SPLITS:
        labels = [_sample_label(item) for item in splits[split]]
        print(
            f"{split}: {len(labels)} frames, {sum(labels)} polyp, {len(labels) - sum(labels)} normal"
        )
    train_ds = PolypDataset(splits["train"], size, True, seed)
    val_ds, test_ds = (
        PolypDataset(splits["val"], size),
        PolypDataset(splits["test"], size),
    )
    sample_ds = PolypDataset(splits["train"], size)
    _plot_grid(
        [sample_ds[i] for i in _preview_indices(splits["train"])],
        output / "samples.png",
        "Training samples after preprocessing",
        show,
    )
    _plot_grid(
        [train_ds[i] for i in _preview_indices(splits["train"])],
        output / "augmented.png",
        "Synchronized training augmentation",
        show,
    )
    generator = torch.Generator().manual_seed(seed)
    labels = [_sample_label(item) for item in splits["train"]]
    counts = {label: labels.count(label) for label in (0, 1)}
    sampler = None
    if config.get("balanced_sampling", True) and all(counts.values()):
        sampler = WeightedRandomSampler(
            [1.0 / counts[label] for label in labels],
            len(labels),
            True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=0,
    )
    val_loader, test_loader = (
        DataLoader(val_ds, batch_size=batch),
        DataLoader(test_ds, batch_size=batch),
    )
    model = TinyUNet(3, base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, float | None]] = []
    best_dice, best_epoch = -1.0, 0
    checkpoint_path = output / "best_model.pt"
    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, val, _, _ = _evaluate(model, val_loader, device)
        dice = float(val["positive_mean_dice"])
        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_dice": dice,
            "val_iou": float(val["positive_mean_iou"]),
            "val_all_frame_dice": float(val["dice"]),
            "val_negative_fp_rate": val["negative_false_positive_frame_rate"],
        }
        history.append(row)
        fp_rate = row["val_negative_fp_rate"]
        fp_text = "n/a" if fp_rate is None else f"{fp_rate:.4f}"
        print(
            f"Epoch {epoch + 1}/{epochs} train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"positive_Dice={dice:.4f} positive_IoU={row['val_iou']:.4f} "
            f"normal_frame_FPR={fp_text}"
        )
        if dice > best_dice:
            best_dice, best_epoch = dice, epoch + 1
            _save_checkpoint(
                checkpoint_path, model, size, base, seed, best_epoch, best_dice
            )
        _save_history(history, output)
    _plot_history(history, output / "history.png", show)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, metrics, previews, rows = _evaluate(
        model, test_loader, device, include_previews=True
    )
    result = {
        "best_epoch": best_epoch,
        "best_val_positive_dice": best_dice,
        "test_loss": test_loss,
        **metrics,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "metric_image_size": size,
        "negative_frame_rule": "Any predicted pixel counts as FP; no 100-pixel component gate.",
    }
    _write_json(output / "test_metrics.json", result)
    _save_test_rows(output / "test_per_image_metrics.csv", root, splits["test"], rows)
    _plot_test(splits["test"], previews, size, output / "test_predictions.png", show)
    print(
        f"Best epoch {best_epoch}; positive test Dice={metrics['positive_mean_dice']:.4f}, "
        f"IoU={metrics['positive_mean_iou']:.4f}; normal frame FPR={metrics['negative_false_positive_frame_rate']}"
    )
    if config.get("inference_image"):
        result["inference"] = predict(
            checkpoint_path, config["inference_image"], output, str(device)
        )
    result["output_dir"] = str(output)
    return result


def _load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[TinyUNet, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TinyUNet(**checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
) -> dict[str, str]:
    """Predict one image, resize the mask to original dimensions, and save mask/overlay."""
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Inference image does not exist: {source}")
    device_obj = _device(device)
    model, checkpoint = _load_checkpoint(checkpoint_path, device_obj)
    size = int(checkpoint["preprocessing"]["image_size"])
    with Image.open(source) as file:
        original = file.convert("RGB")
    resized = original.resize((size, size), Image.Resampling.BILINEAR)
    x = (
        torch.from_numpy(
            np.asarray(resized, dtype=np.float32).transpose(2, 0, 1).copy() / 255
        )
        .unsqueeze(0)
        .to(device_obj)
    )
    probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    full_probs = np.asarray(
        Image.fromarray(probs.astype(np.float32)).resize(
            original.size, Image.Resampling.BILINEAR
        )
    )
    mask = Image.fromarray(
        (full_probs >= checkpoint.get("threshold", 0.5)).astype(np.uint8) * 255, "L"
    )
    mask_array = np.asarray(mask)
    overlay = np.asarray(original).copy()
    foreground = mask_array > 0
    overlay[foreground] = (
        (0.55 * overlay[foreground] + 0.45 * np.array([255, 0, 0]))
        .clip(0, 255)
        .astype(np.uint8)
    )
    output = _prepare_output(output_dir, allow_existing=True)
    mask_path, overlay_path = (
        output / f"{source.stem}_pred_mask.png",
        output / f"{source.stem}_overlay.png",
    )
    if mask_path.exists() or overlay_path.exists():
        raise FileExistsError(f"Inference output already exists in {output}.")
    mask.save(mask_path)
    Image.fromarray(overlay).save(overlay_path)
    print(f"Saved mask: {mask_path}\nSaved overlay: {overlay_path}")
    return {"mask": str(mask_path), "overlay": str(overlay_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a small binary polyp segmentation model or predict one image."
    )
    parser.add_argument("--data-root")
    parser.add_argument("--image-dir")
    parser.add_argument("--mask-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest-csv")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--allow-random-split", action="store_true")
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    config = dict(CONFIG)
    for name in (
        "data_root",
        "image_dir",
        "mask_dir",
        "output_dir",
        "manifest_csv",
        "epochs",
        "image_size",
        "batch_size",
        "base_channels",
        "seed",
        "device",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.no_show:
        config["show_plots"] = False
    if args.allow_random_split:
        config["allow_random_split"] = True
    if args.mode == "infer":
        checkpoint = args.checkpoint or str(
            Path(config["output_dir"]) / "best_model.pt"
        )
        image = args.input or config.get("inference_image")
        if not image:
            raise ValueError("Inference needs --input or CONFIG['inference_image'].")
        return predict(checkpoint, image, config["output_dir"], config["device"])
    return train(config)


def _in_ipython() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


if __name__ == "__main__":
    if _in_ipython():
        train(
            dict(CONFIG)
        )  # Pasted cells use CONFIG and ignore the kernel's -f argument.
    else:
        main()
