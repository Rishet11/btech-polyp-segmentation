# FAR-POLYP-SEG starter

Polyp segmentation on colonoscopy images. Open [`kaggle_polyp.ipynb`](kaggle_polyp.ipynb) in Kaggle for separate preprocessing, training and inference sections. It displays samples, preprocesses images and masks, augments training images, trains a small U-Net, and saves predictions.

## Dataset and links

This project uses **FAR-POLYP-SEG**.

- [Dataset and licence on Zenodo](https://doi.org/10.5281/zenodo.20284781)
- [FAR-POLYP-SEG paper](https://doi.org/10.1007/s10278-026-02268-5)
- [Authors' training code and benchmark splits](https://github.com/FAR-POLYP-SEG/Codes)
- This repository: https://github.com/Rishet11/btech-polyp-segmentation

The release contains 432 polyp images and 7,749 normal images from 455 patients. It uses the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/). Credit the dataset authors and cite the paper in your project report. The data and trained weights are not included here.

Download `Dataset.zip` from Zenodo. The release checksum is `md5:300e09ef379f969a75e8dfa45d64a3cd`. Extract it and upload the extracted data as a Kaggle dataset, then attach that dataset to your notebook. Do not treat a Kaggle upload address as a local file path.

The folder you pass as `data_root` must directly contain the patient folders:

```text
Dataset/
  Patient1/
    Healthy/       # Normal images. The loader creates zero masks.
    Polyp/         # Polyp images.
    BinaryMask/    # Matching masks, with the same filenames as Polyp/.
    Annotated/     # Viewing overlays. The loader does not use these.
  Patient2/
  ...
```

## Run directly in Kaggle

1. Download `kaggle_polyp.ipynb` and import it into Kaggle as a notebook. Attach the extracted FAR dataset and enable a GPU if available.
2. Run **Setup** once. It loads the pipeline functions without starting training.
3. In **1. Preprocessing**, edit `DATA_ROOT` to the attached folder containing `Patient1`, `Patient2`, etc. For example, `/kaggle/input/YOUR-DATASET-SLUG/Dataset`. This is a placeholder, not a verified mount path. Set a new `OUTPUT_DIR`, such as `/kaggle/working/far_run_01`, then run the cell to display images, masks and augmented samples.
4. Run **2. Training** to train and evaluate. Run **3. Inference** to load the checkpoint and predict an image. `INPUT_IMAGE` is optional; the default uses a held-out polyp frame.

The notebook is self-contained. It needs no repository clone after importing. For a single-cell version, paste all of [`kaggle_polyp.py`](kaggle_polyp.py) into a code cell and edit its `CONFIG` dictionary instead.

If you cannot find the root, run this separate cell first. It prints candidates so you can inspect and choose one:

```python
from pathlib import Path
for patient in sorted(Path('/kaggle/input').rglob('Patient1')):
    if patient.is_dir():
        print(patient.parent)
```

These are the fields to edit. **No download link or API key goes in the training code.** Kaggle supplies the attached files under `/kaggle/input`.

| Field or location | What goes there |
| --- | --- |
| Notebook `DATA_ROOT`, `OUTPUT_DIR`, `INPUT_IMAGE` | Local paths for the three stages. The first stage copies the dataset and result paths into `CONFIG` and saves a patient split for training. |
| Kaggle notebook's attached input | Your uploaded FAR dataset. Its URL is for sharing the data input with your team. |
| `CONFIG['data_root']` | Actual local folder containing `PatientN` directories. Required. |
| `CONFIG['output_dir']` | New folder under `/kaggle/working` for results. |
| `CONFIG['inference_image']` | Optional local path to one image. Leave `None` to train and evaluate only. |
| `CONFIG['manifest_csv']` | Leave `None` for the original FAR folders. Set a local CSV path only to reuse a prepared split. |
| `image_dir`, `mask_dir` | Used only for generic paired-image demos. Leave them alone for FAR. |
| Dataset, paper, and repository links above | Use these in the project report. Replace the repository link if you fork it. |

The default run uses 256 x 256 inputs, five epochs, batch size 8, and a U-Net trained from scratch. It needs no pretrained-weight download or extra segmentation library. Five epochs are a starting check, not a performance target. Reduce the batch size if GPU memory runs out.

## What the pipeline does

- Matches `Polyp/` images to `BinaryMask/` files within each patient. Only explicitly labelled `Healthy/` images receive empty masks. Missing positive masks cause an error.
- Splits **patients**, approximately 80%/10%/10%, into training, validation, and test sets. It separates patients with and without polyp images before splitting. A patient's normal and polyp frames stay together. At least three polyp-positive patients are required.
- Converts images to RGB, resizes with bilinear interpolation, and scales to `[0, 1]`. It thresholds JPEG masks at `> 127` and resizes binary masks with nearest-neighbour interpolation. Lossless 0/1 masks also work. If resizing removes an entire polyp mask, it stops and asks for a larger input size.
- Applies matching flips and quarter-turn rotations to each training image and its mask. Brightness and contrast changes affect only the image. Validation and test inputs receive no augmentation.
- Uses binary cross-entropy with logits plus soft Dice loss. Balanced sampling prevents the many normal images from overwhelming the training batches. Validation and test sets retain their original class proportions.
- Saves the checkpoint with the best **positive-image validation Dice**, reloads it, and evaluates the test set once. It reports positive-image Dice/IoU and frame sensitivity separately from false-positive rate and predicted foreground area on normal images.
- Restores inference probabilities to the original image size before thresholding, then saves a binary mask and red overlay.

Dice and intersection over union (IoU) measure overlap with the true mask; higher is better. Here, a normal frame is a false positive if **any pixel** reaches probability 0.5. Frame sensitivity means that a polyp-positive image has any predicted foreground; it does not prove the predicted region overlaps the polyp. A split without normal images has an undefined false-alarm rate, saved as `null` in JSON. Per-image results let you inspect these cases.

Validation and test metrics use the configured input resolution. The authors evaluate at native resolution and also use a connected-component area gate for false alarms. Their published scores are not directly comparable to this starter. Our split and training on normal images also differ from their five-fold protocol. This is a baseline for later calibration experiments, not a reproduction of the paper or evidence of a new research contribution.

## Run from a terminal

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python kaggle_polyp.py \
  --data-root /path/to/Dataset \
  --output-dir outputs/far_run_01 \
  --epochs 5 --batch-size 8 --no-show
```

Run inference from a saved checkpoint:

```bash
python kaggle_polyp.py --mode infer \
  --checkpoint outputs/far_run_01/best_model.pt \
  --input /path/to/new_image.jpg \
  --output-dir outputs/prediction_01 --no-show
```

Or, after running the training cell in Kaggle:

```python
predict(
    '/kaggle/working/far_run_01/best_model.pt',
    '/kaggle/input/YOUR-DATASET-SLUG/Dataset/Patient1/Polyp/YOUR-IMAGE.jpg',
    '/kaggle/working/prediction_01',
)
```

Replace both `YOUR-...` placeholders. Inference uses the model size and preprocessing stored in the checkpoint.

## Saved files

Each run saves `config.json` with package versions, `split.csv`, `samples.png`, `augmented.png`, `history.csv`, `history.json`, `history.png`, `best_model.pt`, `test_metrics.json`, `test_per_image_metrics.csv`, and `test_predictions.png`. Optional inference saves `*_pred_mask.png` and `*_overlay.png`. Existing results are protected from accidental overwrite.

To reuse a split, set `manifest_csv` to the saved `split.csv`. Its columns are `image,mask,split,group,label`; paths are relative to `data_root`. `group` is the actual `PatientN` identifier, not a `groupN` token in a filename. Normal rows have `label=normal` and an empty mask field. Patient groups must not cross splits.

## Checks and limits

Run the focused checks with `python -m unittest discover -s tests -v`. They cover patient separation, missing masks, normal labels, augmentation alignment, metric edge cases, and a one-epoch checkpoint/inference run using synthetic images. A synthetic check proves that the pipeline runs; it says nothing about medical accuracy.

Local verification passed all eight checks. The downloaded release matched the checksum above, and the loader read all 8,181 frames from 455 patients. A separate one-epoch CPU run used 205 real frames from 12 patients, 64 x 64 inputs, and four base channels; checkpoint reload and 672 x 480 inference worked. That quick run marked all 16 normal test images as foreground, so it is not a useful trained model. The default five-epoch Kaggle GPU run remains unverified.

The notebook embeds the script definitions and runs the three stages separately. Its cells passed a separate CPU check with 15 synthetic frames from nine patients: previews ran before training, one epoch saved a checkpoint, and inference restored the original image size. This checks execution, not medical accuracy. To refresh the notebook after changing `kaggle_polyp.py`, run `python tools/build_kaggle_notebook.py`.

This project is for research and coursework. A separate hospital dataset is still needed to test the cross-hospital research question. Keep test patients out of later threshold tuning, and report how many normal images were used for calibration. Do not use this model for clinical decisions.

Cite Lalinia et al., *FAR-POLYP-SEG: A Prospective Single-Center Colonoscopy Dataset for Colorectal Polyp Segmentation with Patient-Level Metadata and Baseline Cross-Dataset Evaluation*, Journal of Imaging Informatics in Medicine (2026), [doi:10.1007/s10278-026-02268-5](https://doi.org/10.1007/s10278-026-02268-5), and the [Zenodo dataset record](https://doi.org/10.5281/zenodo.20284781).
