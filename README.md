# Calibrated Object Measurement Project

End-to-end computer vision pipeline for the XIS AI / Computer Vision technical assessment. The system calibrates a camera, segments a phone cover from images, and (in Step 3) will compute real-world metric measurements in millimetres.

## Project Status

| Step | Phase | Status |
|------|-------|--------|
| Step 1 | Camera Calibration & Data Collection | Complete |
| Step 2 | Model Training & Segmentation | Complete |
| Step 3 | Pixel-to-MM Measurement | Not started |

## Completed Capabilities

- ChArUco board generation and intrinsic camera calibration (22 images)
- Dataset undistortion pipeline for raw object images
- 90-image phone cover dataset with COCO instance segmentation labels
- Train/val/test split (70% / 20% / 10%)
- Mask R-CNN training (15 epochs, Google Colab GPU)
- Test-set evaluation with mAP, IoU, precision, recall, and F1
- Inference script with undistortion and annotated mask output

## Repository Structure

```text
calibration/           # ChArUco board, calibration scripts, images, results
dataset/               # Raw images, undistorted images, train/val/test splits
models/                # Training notebook, config, weights, evaluation
inference/             # Inference script and demo outputs
measurement/           # Step 3 — pixel-to-mm pipeline (pending)
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

## Documentation

| Document | Description |
|----------|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Environment setup and run instructions |
| [docs/CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md) | Camera calibration method and results |
| [docs/DATASET_CARD.md](docs/DATASET_CARD.md) | Dataset description and statistics |
| [docs/TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | Model architecture, training, and metrics |
| [docs/MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md) | Step 3 measurement methodology (pending) |

## Key Artifacts

- Calibration board: `calibration/charuco_board.png`
- Camera matrix: `calibration/results/camera_matrix.npy`
- Model weights: `models/maskrcnn.pth` (local, gitignored)
- Training notebook: `models/mask_rcnn.ipynb`
- Evaluation metrics: `models/evaluation_metrics.json`
