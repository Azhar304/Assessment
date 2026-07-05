# Model Training Report

## Objective

Train an image segmentation model that detects the phone cover and produces a pixel-level mask. The mask is required by Step 3 because metric measurement is computed from the mask contour.

## Model Selection

| Item | Value |
|------|-------|
| Architecture | Mask R-CNN with ResNet-50-FPN backbone |
| Framework | PyTorch / TorchVision |
| Model constructor | `maskrcnn_resnet50_fpn` |
| Pretraining | COCO pretrained weights during training |
| Number of classes | 2: background + `Phone_Cover` |
| Training environment | Google Colab GPU |

## Selection Rationale

Mask R-CNN was selected because:

- The assessment requires segmentation, not only object detection.
- It outputs instance masks suitable for contour extraction and measurement.
- ResNet-50-FPN is a proven baseline for small and medium segmentation datasets.
- TorchVision provides a reproducible implementation independent of Roboflow and Ultralytics YOLO models, which are excluded by the assessment.
- Transfer learning from COCO reduces the amount of custom data required.

## Training Data

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 62 | 62 |
| Validation | 19 | 19 |
| Test | 9 | 9 |

Dataset format:

```text
COCO instance segmentation
```

Foreground class:

```text
Phone_Cover
```

## Training Configuration

Configuration source:

```text
models/config.py
```

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning rate | 0.005 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Scheduler | StepLR |
| Scheduler step size | 10 |
| Scheduler gamma | 0.1 |
| Epochs | 15 |
| Train batch size | 2 |
| Validation batch size | 1 |
| Test batch size | 1 |
| Confidence threshold | 0.5 |
| Mask threshold | 0.5 |
| Random seed | 42 |

## Augmentation Strategy

No explicit online augmentation was applied during training. The dataset includes real capture variation in object rotation, background, scale, angle, and lighting. Future work should add light geometric and photometric augmentation to improve robustness.

## Training Pipeline

Artifacts:

| Artifact | Path |
|----------|------|
| Training notebook | `models/mask_rcnn.ipynb` |
| Configuration | `models/config.py` |
| Shared model utilities | `models/model_utils.py` |
| Loss history | `models/training_history.json` |
| Best weights | `models/maskrcnn.pth` |

Training steps:

1. Mount Google Drive in Colab.
2. Load the COCO train and validation splits.
3. Convert polygon annotations to binary masks.
4. Build Mask R-CNN with custom classification and mask heads.
5. Train for 15 epochs.
6. Track train and validation loss.
7. Save the best checkpoint by validation loss.
8. Visualise predictions on held-out test images.

## Loss History

| Epoch | Train Loss | Validation Loss |
|-------|------------|-----------------|
| 1 | 0.9400 | 0.8159 |
| 5 | 0.1286 | 0.1136 |
| 10 | 0.0706 | 0.0919 |
| 13 | 0.0635 | 0.0874 |
| 15 | 0.0622 | 0.0883 |

Summary:

| Metric | Value |
|--------|-------|
| Epochs completed | 15 |
| Final train loss | 0.0622 |
| Final validation loss | 0.0883 |
| Best validation loss | 0.0874 |
| Best validation epoch | 13 |

Numeric history is saved in:

```text
models/training_history.json
```

## Test Evaluation

Evaluation script:

```text
models/evaluate.py
```

Command:

```powershell
python models/evaluate.py
```

Metrics file:

```text
models/evaluation_metrics.json
```

| Metric | Value |
|--------|-------|
| Test images | 9 |
| mAP@0.5 | 1.0000 |
| mAP@0.5:0.95 | 0.9523 |
| Mean mask IoU | 0.9565 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| True positives | 9 |
| False positives | 0 |
| False negatives | 0 |

All 9 test images produced a correct phone-cover prediction above the confidence threshold. The mean predicted mask IoU against test labels is 0.9565.

## Inference Pipeline

Inference script:

```text
inference/run_inference.py
```

Command:

```powershell
python inference/run_inference.py path/to/raw_image.jpg
```

Pipeline:

1. Read the input image.
2. Undistort it using `calibration/results/camera_matrix.npy` and `dist_coeffs.npy`.
3. Load `models/maskrcnn.pth`.
4. Run Mask R-CNN inference.
5. Apply confidence and mask thresholds.
6. Draw mask overlay, bounding box, and confidence label.
7. Save the undistorted and annotated images.

Outputs:

```text
inference/outputs/<name>_undistorted.jpg
inference/outputs/<name>_annotated.jpg
```

Example output:

```text
Input image       : dataset/raw/images/IMG20260704120145.jpg
Undistorted saved : inference/outputs/IMG20260704120145_undistorted.jpg
Annotated saved   : inference/outputs/IMG20260704120145_annotated.jpg
Detections        : 1
  [1] Phone_Cover conf=0.989 bbox=[798, 604, 2046, 2191]
```

## Module Interfaces

| Function | Location | Purpose |
|----------|----------|---------|
| `build_model(num_classes=2)` | `models/model_utils.py` | Build Mask R-CNN with custom heads. |
| `load_model(weights_path=None, device=None)` | `models/model_utils.py` | Load trained weights and return model/device. |
| `undistort_image(image, camera_matrix=None, dist_coeffs=None)` | `models/model_utils.py` | Apply OpenCV undistortion and ROI crop. |
| `run_inference(image_path, output_dir, weights_path=None, skip_undistort=False)` | `inference/run_inference.py` | Run end-to-end image inference. |

## Reproducibility

- Dataset split seed: 42.
- Training hyperparameters are recorded in `models/config.py`.
- Evaluation thresholds are fixed at confidence 0.5 and mask 0.5.
- Metrics are saved in JSON for repeatable reporting.
- The training notebook documents the Colab workflow.

## Limitations

- The test set contains only 9 images, so metrics should be interpreted carefully.
- No online augmentation was used.
- Google Colab hardware may vary between runs.
- The current camera calibration RMS error is 1.008605 px, above the recommended 0.5 px threshold.
- The trained weights file is large and may need to be submitted outside Git.

## Requirements Checklist

| Requirement | Status |
|-------------|--------|
| Non-YOLO, non-Roboflow architecture | Met |
| Architecture justified | Met |
| 70/20/10 split | Met |
| Hyperparameters documented | Met |
| Train/validation losses logged | Met |
| mAP@0.5 reported | Met |
| mAP@0.5:0.95 reported | Met |
| IoU, precision, recall, F1 reported | Met |
| Inference script provided | Met |
| Annotated prediction output | Met |
