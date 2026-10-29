"""Build the self-contained notebook from the tested pipeline definitions."""

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, cell_id, **metadata):
    result = {
        "cell_type": kind,
        "id": cell_id,
        "metadata": metadata,
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    source = (ROOT / "kaggle_polyp.py").read_text()
    entrypoint = ast.parse(source).body[-1]
    if not isinstance(entrypoint, ast.If) or ast.unparse(entrypoint.test) != "__name__ == '__main__'":
        raise ValueError("Expected the script's final __main__ entrypoint.")
    definitions = "".join(source.splitlines(keepends=True)[: entrypoint.lineno - 1])
    cells = [
        cell("markdown", """
        # FAR-POLYP-SEG in Kaggle

        Attach the extracted [FAR dataset](https://doi.org/10.5281/zenodo.20284781)
        with **Add Input** and enable a GPU in notebook settings. Run the setup
        cell once, then the three sections below in order. Only the marked path
        fields need editing. No download URL or API key belongs in a path field.

        Section 1 displays images and masks before any training, so it can be
        used alone for a data preview. Section 2 trains and evaluates the model.
        Section 3 loads the saved checkpoint and predicts one image.
        """, "intro"),
        cell("markdown", """
        ## Setup

        This cell contains the same definitions as `kaggle_polyp.py`, with its
        automatic training entrypoint removed. It loads functions but does not
        train. Kaggle's Python environment supplies PyTorch, Pillow, NumPy and
        Matplotlib. No repository clone or internet access is needed after you
        attach the dataset.
        """, "setup-notes"),
        cell("code", definitions, "setup", jupyter={"source_hidden": True}),
        cell("markdown", """
        ## 1. Preprocessing, image display and augmentation

        Set `DATA_ROOT` to the folder that directly contains `Patient1`,
        `Patient2`, etc. Set a new `OUTPUT_DIR` for each training run.
        The patient split is saved in a separate preview folder and reused in
        training. Only training images receive augmentation. Images use RGB,
        bilinear resizing and values in [0, 1]. Masks use binary thresholding
        and nearest-neighbour resizing. Healthy frames receive empty masks.
        """, "preprocessing-notes"),
        cell("code", """
        DATA_ROOT = "/kaggle/input/YOUR-DATASET-SLUG/Dataset"  # EDIT: attached FAR folder
        OUTPUT_DIR = "/kaggle/working/far_run_01"  # EDIT: new result folder per run

        CONFIG.update(data_root=DATA_ROOT, output_dir=OUTPUT_DIR, inference_image=None)
        # Optional: change CONFIG["epochs"], CONFIG["batch_size"] or CONFIG["image_size"].
        splits = discover_far_polyp(
            DATA_ROOT, CONFIG["seed"], CONFIG["train_fraction"],
            CONFIG["val_fraction"], CONFIG["test_fraction"],
        )
        if splits is None:
            raise ValueError("DATA_ROOT must directly contain the FAR PatientN folders.")
        preview_dir = _prepare_output(f"{OUTPUT_DIR}_preview", allow_existing=True)
        manifest = preview_dir / "split.csv"
        _write_splits(manifest, _root(DATA_ROOT), splits)
        CONFIG["manifest_csv"] = str(manifest)

        for name, rows in splits.items():
            positive = sum(row.label == "polyp" for row in rows)
            print(f"{name}: {len(rows)} frames, {positive} polyp, {len(rows) - positive} normal")
        print(f"Saved patient split: {manifest}")

        plain = PolypDataset(splits["train"], CONFIG["image_size"])
        augmented = PolypDataset(splits["train"], CONFIG["image_size"], True, CONFIG["seed"])
        indices = _preview_indices(splits["train"])
        _plot_grid([plain[i] for i in indices], preview_dir / "samples.png",
                   "Images and masks after preprocessing", CONFIG["show_plots"])
        _plot_grid([augmented[i] for i in indices], preview_dir / "augmented.png",
                   "Matching image and mask augmentation", CONFIG["show_plots"])
        """, "preprocessing"),
        cell("markdown", """
        ## 2. Training and held-out test evaluation

        The default is five epochs at 256 x 256. Reduce the batch size if GPU
        memory runs out. This stage uses the saved patient split, selects a
        checkpoint on positive validation Dice, reloads it, then tests once.
        Normal-frame false alarms and positive-mask overlap are reported
        separately. A short run checks the code; it does not establish accuracy.
        Existing training results are protected. Choose a new `OUTPUT_DIR` in
        section 1 before running another experiment.
        """, "training-notes"),
        cell("code", """
        training_result = train(dict(CONFIG))
        print(f"Saved checkpoint: {Path(OUTPUT_DIR) / 'best_model.pt'}")
        """, "training"),
        cell("markdown", """
        ## 3. Inference from the saved checkpoint

        Leave `INPUT_IMAGE` as `None` to preview a held-out polyp image, or set
        it to another local image path. Set a new `INFERENCE_DIR` to repeat this
        stage without overwriting predictions. The checkpoint supplies the
        model settings. Predictions are restored to the original image size.
        """, "inference-notes"),
        cell("code", """
        INPUT_IMAGE = None  # EDIT: optional local image path, not a URL
        INFERENCE_DIR = f"{OUTPUT_DIR}_inference"  # EDIT: new folder for another prediction
        if INPUT_IMAGE is None:
            INPUT_IMAGE = str(next(row.image for row in splits["test"] if row.label == "polyp"))
        prediction = predict(
            Path(OUTPUT_DIR) / "best_model.pt", INPUT_IMAGE,
            INFERENCE_DIR, device=CONFIG["device"],
        )
        print(json.dumps(prediction, indent=2))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, path, title in zip(
            axes, [INPUT_IMAGE, prediction["mask"], prediction["overlay"]],
            ["Input", "Predicted mask", "Overlay"],
        ):
            with Image.open(path) as image:
                axis.imshow(image.copy(), cmap="gray" if title == "Predicted mask" else None)
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        if CONFIG["show_plots"]:
            plt.show()
        plt.close(fig)
        """, "inference"),
        cell("markdown", """
        ## Results and source

        `OUTPUT_DIR` contains the split, checkpoint, plots, training history,
        test metrics and per-image test results. Section 3 saves a mask and
        overlay in `INFERENCE_DIR`. See the repository README for metric rules
        and differences from the authors' evaluation protocol.

        Cite the [FAR-POLYP-SEG paper](https://doi.org/10.1007/s10278-026-02268-5)
        and [dataset release](https://doi.org/10.5281/zenodo.20284781).
        The dataset uses CC BY 4.0. This notebook is for research and coursework;
        its predictions are not suitable for clinical decisions.
        """, "results"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "kaggle_polyp.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
