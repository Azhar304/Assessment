# Model Artifacts

This folder contains the Mask R-CNN training notebook, shared model utilities, configuration, trained weights, and evaluation artifacts for the phone-cover segmentation model.

## Architecture

| Item | Value |
|------|-------|
| Model | Mask R-CNN |
| Backbone | ResNet-50-FPN |
| Framework | PyTorch / TorchVision |
| Classes | background + `Phone_Cover` |
| Output | Bounding boxes, confidence scores, and instance masks |

Mask R-CNN was selected because the measurement step needs a pixel-level object mask. Roboflow models and Ultralytics YOLO models were not used.

## Dataset

Training uses COCO-style instance segmentation data:

```text
dataset/train/images/ + annotations.json
dataset/val/images/   + annotations.json
dataset/test/images/  + annotations.json
```

Split counts:

```text
Train:      62 images
Validation: 19 images
Test:        9 images
```

## Training Configuration

Full configuration is recorded in:

```text
models/config.py
```

Key values:

```text
Optimizer:            SGD
Learning rate:        0.005
Momentum:             0.9
Weight decay:         0.0005
Scheduler:            StepLR
Scheduler step/gamma: 10 / 0.1
Epochs:               15
Train batch size:     2
Validation batch:     1
Confidence threshold: 0.5
Mask threshold:       0.5
Random seed:          42
```

## Training Results

Training was performed in Google Colab. Best validation loss occurred at epoch 13.

```text
Final train loss:      0.0622
Final validation loss: 0.0883
Best validation loss:  0.0874
Best epoch:            13
```

Loss history:

```text
models/training_history.json
```

## Test Metrics

Run:

```powershell
python models/evaluate.py
```

Current results:

```text
Test images:       9
mAP@0.5:           1.0000
mAP@0.5:0.95:      0.9523
Mean mask IoU:     0.9565
Precision:         1.0000
Recall:            1.0000
F1:                1.0000
True positives:    9
False positives:   0
False negatives:   0
```

Full metrics:

```text
models/evaluation_metrics.json
```

## Scripts and Modules

| File | Purpose |
|------|---------|
| `mask_rcnn.ipynb` | Google Colab training notebook. |
| `config.py` | Reproducible training and evaluation constants. |
| `model_utils.py` | Shared model loading and image undistortion helpers. |
| `evaluate.py` | Test-set COCO/mask evaluation. |

Important functions:

| Function | Purpose |
|----------|---------|
| `build_model(num_classes=2)` | Create Mask R-CNN with custom heads. |
| `load_model(weights_path=None, device=None)` | Load trained weights for inference/evaluation. |
| `undistort_image(image, camera_matrix=None, dist_coeffs=None)` | Apply calibration-based undistortion. |

## Weights

| File | Description |
|------|-------------|
| `models/maskrcnn.pth` | Best trained weights. |

The weights file is large and may be ignored by Git. If it is not included in the repository submission, provide it separately through Google Drive or another artifact link.

## Limitations

- The test set is small.
- No explicit online augmentation was used.
- Training hardware in Colab may vary.
- Model performance depends on calibrated/undistorted images matching the training distribution.
