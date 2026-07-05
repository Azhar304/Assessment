# Calibrated Object Measurement Project

End-to-end computer vision pipeline for the XIS AI / Computer Vision technical assessment. The project calibrates a camera, prepares a self-collected segmentation dataset, trains a Mask R-CNN model to segment a phone cover, and converts the predicted mask into real-world width and height measurements in millimetres.

## Assessment Coverage

| Step | Phase | Status |
|------|-------|--------|
| Step 1 | Camera Calibration and Data Collection | Complete |
| Step 2 | Model Training and Segmentation | Complete |
| Step 3 | Pixel-to-MM Measurement | Complete with 10-sample validation |

## Key Capabilities

- ChArUco board generation and OpenCV intrinsic camera calibration.
- 22 calibration images captured from varied angles, distances, and board positions.
- Camera matrix and distortion coefficients saved in `calibration/results/`.
- Dataset undistortion pipeline for raw phone-cover images.
- 92 raw object images, 92 undistorted images, and 90 labelled COCO instance-segmentation samples.
- Train/validation/test split close to the required 70/20/10 target: 62/19/9 images.
- Mask R-CNN with ResNet-50-FPN backbone trained for 15 epochs in Google Colab.
- Test-set evaluation with mAP, IoU, precision, recall, and F1.
- Single-image inference script with undistortion and mask overlay output.
- Pixel-to-mm measurement pipeline using an automatically detected ISO/IEC 7810 ID-1 card reference.
- 10-image physical validation report with MAE and MPE.

## Object Selection

The selected target object is a phone cover with approximate physical dimensions of 80 mm width and 170 mm height. It was chosen because it is readily available, mostly planar, easy to label as one connected region, and has a clear rectangular shape suitable for width/height measurement.

## System Architecture

```text
calibration images
  -> ChArUco corner detection
  -> intrinsic camera calibration
  -> camera_matrix.npy + dist_coeffs.npy

raw phone-cover images
  -> cv2.undistort using calibration parameters
  -> CVAT polygon labelling
  -> COCO dataset split
  -> Mask R-CNN training and evaluation

new measurement image
  -> undistortion
  -> reference card detection
  -> pixels_per_mm calculation
  -> Mask R-CNN segmentation
  -> mask contour minimum-area rectangle
  -> width_mm, height_mm, confidence, annotated output
```

## Repository Structure

```text
calibration/           ChArUco board, calibration scripts, calibration images, results
dataset/               Raw images, undistorted images, COCO labels, train/val/test splits
models/                Training notebook, model config, weights, evaluation utilities
inference/             Single-image inference script and demo outputs
measurement/           Pixel-to-mm measurement pipeline and validation artifacts
docs/                  Required technical documentation
requirements.txt       Python dependencies
README.md              Project overview and quick start
```

## Quick Start

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run inference on a raw image:

```powershell
python inference/run_inference.py dataset/raw/images/IMG20260704120145.jpg
```

Evaluate the trained model:

```powershell
python models/evaluate.py
```

Measure a phone cover from a new image containing the phone cover and a visible card-sized reference object:

```powershell
python measurement/measure_object.py measurement/validation_images/01.jpg
```

Validate measurement accuracy from saved predictions:

```powershell
python measurement/validate_measurements.py measurement/validation_results.csv
```

Run the full 10-image validation batch:

```powershell
python measurement/run_validation.py
```

## Main Results

### Calibration

| Metric | Value |
|--------|-------|
| Calibration images | 22 |
| Valid calibration images | 22 / 22 |
| RMS reprojection error | 1.008605 px |
| Camera matrix | `calibration/results/camera_matrix.npy` |
| Distortion coefficients | `calibration/results/dist_coeffs.npy` |

The reprojection error is documented honestly because it is above the assessment guidance target of 0.5 px. Undistortion is still applied throughout inference and measurement.

### Segmentation

| Metric | Value |
|--------|-------|
| Test images | 9 |
| mAP@0.5 | 1.0000 |
| mAP@0.5:0.95 | 0.9523 |
| Mean mask IoU | 0.9565 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |

### Measurement

| Metric | Value |
|--------|-------|
| Validation samples | 10 |
| Width MAE | 2.67 mm |
| Height MAE | 5.22 mm |
| Overall MAE | 3.95 mm |
| Width MPE | 3.34% |
| Height MPE | 3.07% |
| Overall MPE | 3.20% |

## API and Module Summary

| Module | Main function or command | Purpose |
|--------|--------------------------|---------|
| `calibration/generate_charuco_board.py` | `python calibration/generate_charuco_board.py` | Generate the printable ChArUco board image. |
| `calibration/calibrate_camera.py` | `python calibration/calibrate_camera.py` | Estimate intrinsics and distortion coefficients. |
| `calibration/undistort_dataset.py` | `python calibration/undistort_dataset.py` | Undistort raw dataset images before labelling/training. |
| `dataset/prepare_dataset.py` | `python dataset/prepare_dataset.py` | Split COCO dataset into train/val/test folders. |
| `models/evaluate.py` | `python models/evaluate.py` | Evaluate segmentation metrics on the test set. |
| `inference/run_inference.py` | `run_inference(image_path, output_dir, weights_path, skip_undistort)` | Undistort, segment, and save annotated mask output. |
| `measurement/measure_object.py` | `measure_image(image_path, ...)` | End-to-end raw image to metric dimensions. |
| `measurement/validate_measurements.py` | `compute_validation(rows)` | Compute MAE and MPE from validation CSV rows. |
| `measurement/run_validation.py` | `python measurement/run_validation.py` | Run measurement on all validation images and write summary files. |

## Required Documentation

| Document | Covers |
|----------|--------|
| [docs/SETUP.md](docs/SETUP.md) | Installation, environment, command usage, reproducibility steps |
| [docs/CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md) | Calibration method, images, parameters, reprojection error, undistortion |
| [docs/DATASET_CARD.md](docs/DATASET_CARD.md) | Object choice, collection strategy, labelling, splits, class distribution |
| [docs/TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | Architecture, hyperparameters, loss history, metrics, inference usage |
| [docs/MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md) | Pixel-to-mm derivation, reference object, validation table, limitations |

## Design Decisions

- Mask R-CNN was selected because the task requires segmentation masks, not only bounding boxes.
- A ChArUco board was used instead of a plain checkerboard because marker IDs improve detection robustness when the board is tilted or partly visible.
- All object images are undistorted before training and measurement to keep the training distribution consistent with the calibrated measurement pipeline.
- A known ID-1 card reference is auto-detected during measurement to avoid manual reference-box entry and make the demo more end-to-end.
- Minimum-area rectangles are computed from the predicted mask contour so rotated phone covers can still be measured.

## Assumptions and Limitations

- The measurement image must contain the phone cover and a fully visible card-sized reference object in the same plane or nearly the same plane.
- Perspective effects are only partially handled by minimum-area rectangles; strong 3D tilt can still introduce error.
- Calibration RMS error is 1.008605 px, above the recommended threshold. Better calibration images would improve measurement reliability.
- The test set has only 9 images, so reported model metrics are promising but statistically limited.
- Large artifacts such as `models/maskrcnn.pth`, dataset folders, validation images, and generated outputs may be shared outside Git if repository size limits apply.
