# Calibrated Object Measurement Project

End-to-end computer vision pipeline for the XIS AI / Computer Vision technical assessment. The system calibrates a camera, segments a phone cover from images, and computes real-world metric measurements in millimetres.

## Project Status

| Step | Phase | Status |
|------|-------|--------|
| Step 1 | Camera Calibration & Data Collection | Complete |
| Step 2 | Model Training & Segmentation | Complete |
| Step 3 | Pixel-to-MM Measurement | Implemented, physical validation pending |

## Completed Capabilities

- ChArUco board generation and intrinsic camera calibration (22 images)
- Dataset undistortion pipeline for raw object images
- 90-image phone cover dataset with COCO instance segmentation labels
- Train/val/test split (70% / 20% / 10%)
- Mask R-CNN training (15 epochs, Google Colab GPU)
- Test-set evaluation with mAP, IoU, precision, recall, and F1
- Inference script with undistortion and annotated mask output
- Pixel-to-mm measurement script using a known-size reference object
- Measurement validation helper for MAE and MPE

## Repository Structure

```text
calibration/           # ChArUco board, calibration scripts, images, results
dataset/               # Raw images, undistorted images, train/val/test splits
models/                # Training notebook, config, weights, evaluation
inference/             # Inference script and demo outputs
measurement/           # Pixel-to-mm measurement pipeline
docs/                  # All documentation
requirements.txt
README.md
```

## Quick Start

Install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run inference on a new raw image:

```powershell
python inference/run_inference.py dataset/raw/images/IMG20260704120145.jpg
```

Evaluate the model on the test set:

```powershell
python models/evaluate.py
```

Measure a phone cover using a known reference object:

```powershell
python measurement/measure_object.py path/to/image.jpg --reference-bbox x,y,width,height --reference-width-mm 85.6 --reference-height-mm 54.0
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Environment setup and run instructions |
| [docs/CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md) | Camera calibration method and results |
| [docs/DATASET_CARD.md](docs/DATASET_CARD.md) | Dataset description and statistics |
| [docs/TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | Model architecture, training, and metrics |
| [docs/MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md) | Pixel-to-mm measurement methodology and validation plan |

## Key Artifacts

- Calibration board: `calibration/charuco_board.png`
- Camera matrix: `calibration/results/camera_matrix.npy`
- Model weights: `models/maskrcnn.pth` (local, gitignored)
- Training notebook: `models/mask_rcnn.ipynb`
- Evaluation metrics: `models/evaluation_metrics.json`
- Measurement script: `measurement/measure_object.py`
