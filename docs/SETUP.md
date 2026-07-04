# Setup Guide

Full installation, environment, and run instructions for the calibrated object measurement pipeline.

## Environment

- Python 3.10+
- Windows / Linux / macOS
- GPU recommended for training (CPU works for inference, evaluation, and measurement)

## Install Dependencies

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU training in Google Colab, install PyTorch with CUDA support. The local `requirements.txt` lists CPU-compatible versions; adjust torch/torchvision for your CUDA version if needed.

## Step 1 - Camera Calibration

Generate the ChArUco board:

```powershell
python calibration/generate_charuco_board.py
```

Output:

```text
calibration/charuco_board.png
```

Place 20+ calibration images in `calibration/images/`, then run intrinsic calibration:

```powershell
python calibration/calibrate_camera.py
```

Outputs:

```text
calibration/results/camera_matrix.npy
calibration/results/dist_coeffs.npy
```

Undistort calibration images for visual verification:

```powershell
python calibration/undistorted_image.py
```

Output:

```text
calibration/undistorted/
```

## Step 1 - Dataset Preparation

Place raw object images in `dataset/raw/images/`, then undistort them:

```powershell
python calibration/undistort_dataset.py
```

Output:

```text
dataset/undistorted/images/
```

After labelling and exporting COCO annotations to `dataset/undistorted/annotations/instances_default.json`, split the dataset:

```powershell
python dataset/prepare_dataset.py
```

Outputs:

```text
dataset/train/   (70%)
dataset/val/     (20%)
dataset/test/    (10%)
```

## Step 2 - Model Training

Training is performed in Google Colab using `models/mask_rcnn.ipynb`. Mount Google Drive, point paths to the prepared dataset, and run all cells.

After training, download the best weights to:

```text
models/maskrcnn.pth
```

Training history is saved automatically to:

```text
models/training_history.json
```

## Step 2 - Evaluation

Run test-set evaluation (requires `models/maskrcnn.pth`):

```powershell
python models/evaluate.py
```

Outputs metrics to `models/evaluation_metrics.json` and prints mAP, IoU, precision, recall, and F1.

## Step 2 - Inference

Run end-to-end inference on a single raw image:

```powershell
python inference/run_inference.py path/to/image.jpg
```

Optional flags:

```powershell
python inference/run_inference.py path/to/image.jpg --output-dir inference/outputs
python inference/run_inference.py path/to/image.jpg --skip-undistort
python inference/run_inference.py path/to/image.jpg --weights models/maskrcnn.pth
```

Outputs:

```text
inference/outputs/<name>_undistorted.jpg
inference/outputs/<name>_annotated.jpg
```

## Step 3 - Measurement

Run phone-cover measurement with a known-size reference object in the image:

```powershell
python measurement/measure_object.py path/to/image.jpg --reference-bbox x,y,width,height --reference-width-mm 85.6 --reference-height-mm 54.0
```

The reference bbox must be provided in processed image coordinates:

```text
x,y,width,height
```

Use `--skip-undistort` only if the input image is already undistorted.

Outputs:

```text
measurement/outputs/<name>_measurement.jpg
measurement/outputs/<name>_measurement.json
```

Validate 10+ measured samples after filling `measurement/validation_template.csv`:

```powershell
python measurement/validate_measurements.py measurement/validation_template.csv
```

This writes MAE and MPE summary metrics to:

```text
measurement/validation_summary.json
```
