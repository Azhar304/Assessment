# Setup Guide

This guide explains how to reproduce the calibrated object measurement pipeline from environment setup through calibration, dataset preparation, training, inference, and measurement validation.

## Environment

| Item | Requirement |
|------|-------------|
| Python | 3.10 or newer |
| OS | Windows, Linux, or macOS |
| GPU | Recommended for training; optional for inference/evaluation/measurement |
| Main libraries | OpenCV contrib, PyTorch, TorchVision, pycocotools, NumPy, Pillow |

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Google Colab training, install a CUDA-compatible PyTorch build if the default Colab environment does not already provide one.

If a local virtual environment becomes invalid after moving machines, recreate it:

```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Repository Inputs

Expected files and folders:

```text
calibration/images/                         20+ calibration images
calibration/charuco_board.png               Generated board image
calibration/results/camera_matrix.npy       Saved intrinsics
calibration/results/dist_coeffs.npy         Saved distortion coefficients
dataset/raw/images/                         Raw object images
dataset/undistorted/annotations/instances_default.json
models/maskrcnn.pth                         Trained model weights
measurement/validation_images/              10 measurement validation images
measurement/validation_results.csv          Validation predictions and ground truth
```

## Step 1 - Camera Calibration

Generate the ChArUco board:

```powershell
python calibration/generate_charuco_board.py
```

Print or display the board, capture 20+ images from varied angles, and place them in:

```text
calibration/images/
```

Run intrinsic calibration:

```powershell
python calibration/calibrate_camera.py
```

Outputs:

```text
calibration/results/camera_matrix.npy
calibration/results/dist_coeffs.npy
```

Undistort calibration images for visual inspection:

```powershell
python calibration/undistorted_image.py
```

Output:

```text
calibration/undistorted/
```

## Step 1 - Dataset Preparation

Place raw phone-cover images in:

```text
dataset/raw/images/
```

Undistort all raw object images:

```powershell
python calibration/undistort_dataset.py
```

Output:

```text
dataset/undistorted/images/
```

Label the undistorted images in CVAT as polygon instance masks and export COCO 1.0 annotations to:

```text
dataset/undistorted/annotations/instances_default.json
```

Create the train/validation/test split:

```powershell
python dataset/prepare_dataset.py
```

Expected split:

```text
dataset/train/images/ + annotations.json   62 images
dataset/val/images/   + annotations.json   19 images
dataset/test/images/  + annotations.json    9 images
```

## Step 2 - Training

Training is performed in Google Colab using:

```text
models/mask_rcnn.ipynb
```

Training procedure:

1. Upload or mount the prepared `dataset/` folder in Google Drive.
2. Confirm paths in `models/config.py`.
3. Run the notebook cells in order.
4. Train Mask R-CNN for 15 epochs.
5. Save the best model weights as `models/maskrcnn.pth`.
6. Save numeric loss history to `models/training_history.json`.

The trained weights are large and may be excluded from Git. If excluded, submit them separately or provide a download location.

## Step 2 - Evaluation

Run test-set evaluation:

```powershell
python models/evaluate.py
```

Outputs:

```text
models/evaluation_metrics.json
```

Reported metrics include:

- mAP@0.5
- mAP@0.5:0.95
- Mean mask IoU
- Precision
- Recall
- F1
- True positives, false positives, and false negatives

## Step 2 - Inference

Run inference on a raw image:

```powershell
python inference/run_inference.py path/to/raw_image.jpg
```

Optional flags:

```powershell
python inference/run_inference.py path/to/raw_image.jpg --output-dir inference/outputs
python inference/run_inference.py path/to/raw_image.jpg --weights models/maskrcnn.pth
python inference/run_inference.py path/to/undistorted_image.jpg --skip-undistort
```

Outputs:

```text
inference/outputs/<name>_undistorted.jpg
inference/outputs/<name>_annotated.jpg
```

Sample output:

```text
Detections        : 1
  [1] Phone_Cover conf=0.989 bbox=[798, 604, 2046, 2191]
```

## Step 3 - Measurement

The measurement image must contain:

- The phone cover target.
- A visible ISO/IEC 7810 ID-1 card-sized reference object, such as a debit/credit card.

Run measurement:

```powershell
python measurement/measure_object.py measurement/validation_images/01.jpg
```

Optional flags:

```powershell
python measurement/measure_object.py path/to/image.jpg --reference-width-mm 85.60 --reference-height-mm 53.98
python measurement/measure_object.py path/to/image.jpg --weights models/maskrcnn.pth
python measurement/measure_object.py path/to/undistorted_image.jpg --skip-undistort
python measurement/measure_object.py path/to/image.jpg --debug
```

Outputs:

```text
measurement/outputs/<name>_measurement.jpg
measurement/outputs/<name>_measurement.json
```

The JSON output includes:

- Width in pixels and millimetres.
- Height in pixels and millimetres.
- Pixels-per-mm ratio.
- Model confidence score.
- Predicted bounding box.
- Reference detection method and reference box points.
- Whether undistortion was applied.

## Step 3 - Validation

Validate already saved predictions:

```powershell
python measurement/validate_measurements.py measurement/validation_results.csv
```

Run the complete validation batch over all images in `measurement/validation_images/`:

```powershell
python measurement/run_validation.py
```

Outputs:

```text
measurement/validation_results.csv
measurement/validation_summary.json
measurement/outputs/<sample>_measurement.jpg
measurement/outputs/<sample>_measurement.json
```

Current validation summary:

```text
Samples:      10
Width MAE:    2.67 mm
Height MAE:   5.22 mm
Overall MAE:  3.95 mm
Overall MPE:  3.20%
```

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `models/maskrcnn.pth` not found | Weights were not downloaded from Colab | Copy trained weights to `models/maskrcnn.pth` or pass `--weights`. |
| No reference card detected | Card not fully visible or low contrast | Improve lighting/background or run with `--debug`. |
| No prediction above threshold | Model confidence too low or wrong input object | Check input image, weights, and `--confidence-threshold`. |
| Measurement changes with object position | Lens distortion or perspective effects | Ensure undistortion is applied and keep object/reference close to the same plane. |
| `pycocotools` import error on Windows | Package install issue | Reinstall dependencies in a fresh virtual environment. |
