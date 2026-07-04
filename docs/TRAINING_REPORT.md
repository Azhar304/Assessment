# Model Training Report

## Model Selection

| Item | Value |
|------|-------|
| Architecture | Mask R-CNN with ResNet-50-FPN backbone |
| Framework | PyTorch / TorchVision |
| Model name | `maskrcnn_resnet50_fpn` |
| Pretrained weights | COCO `DEFAULT` (transfer learning) |
| Training environment | Google Colab GPU |

### Justification

Mask R-CNN was chosen because:

- It produces pixel-level instance segmentation masks, required for accurate width/height measurement from contour geometry
- The ResNet-50-FPN backbone is a well-established, reproducible baseline with strong COCO pretrained weights
- TorchVision provides a maintained implementation without dependency on Roboflow or Ultralytics YOLO (both excluded by assessment requirements)
- Two-stage detection + segmentation handles single-object scenes reliably with moderate dataset size

## Training Configuration

All hyperparameters are recorded in `models/config.py`:

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning rate | 0.005 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| LR scheduler | StepLR (step size 10, gamma 0.1) |
| Epochs | 15 |
| Train batch size | 2 |
| Validation batch size | 1 |
| Confidence threshold | 0.5 |
| Mask threshold | 0.5 |
| Random seed | 42 |
| Num classes | 2 (background + Phone_Cover) |

### Augmentation

No explicit augmentation transforms were applied in the training loop. Dataset diversity was achieved through varied image capture conditions.

## Training Pipeline

| Artifact | Path |
|----------|------|
| Training notebook | `models/mask_rcnn.ipynb` |
| Configuration | `models/config.py` |
| Loss history | `models/training_history.json` |
| Best weights | `models/maskrcnn.pth` |

Training steps in the notebook:

1. Mount Google Drive and load COCO dataset splits
2. Build `CocoMaskDataset` with polygon-to-mask conversion
3. Initialise Mask R-CNN with COCO pretrained weights and custom heads
4. Train for 15 epochs, saving checkpoints each epoch
5. Save best model by validation loss
6. Plot and save loss curves
7. Visualise predictions on test images

## Loss Curves

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.9400 | 0.8159 |
| 5 | 0.1286 | 0.1136 |
| 10 | 0.0706 | 0.0919 |
| 13 | 0.0635 | **0.0874** (best) |
| 15 | 0.0622 | 0.0883 |

Summary:

| Metric | Value |
|--------|-------|
| Final train loss | 0.0622 |
| Final validation loss | 0.0883 |
| Best validation loss | 0.0874 (epoch 13) |
| Epochs completed | 15 |

Loss curves were saved during Colab training to `loss_curve.png` on Google Drive. The numeric history is in `models/training_history.json`.

## Test Set Evaluation

Evaluated with `models/evaluate.py` using pycocotools COCO segmentation metrics:

| Metric | Value |
|--------|-------|
| mAP@0.5 | 1.0000 |
| mAP@0.5:0.95 | 0.9523 |
| Mean mask IoU | 0.9565 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| True positives | 9 |
| False positives | 0 |
| False negatives | 0 |

Full metrics saved to `models/evaluation_metrics.json`.

All 9 test images were detected with a single correct Phone_Cover prediction. Segmentation masks align closely with ground truth (mean IoU 95.65%).

## Inference Pipeline

Implemented in `inference/run_inference.py`.

Pipeline flow:

1. Load raw input image
2. Undistort using calibration parameters from `calibration/results/`
3. Run Mask R-CNN inference
4. Overlay segmentation mask, bounding box, and confidence label
5. Save undistorted and annotated output images

### Usage

```powershell
python inference/run_inference.py path/to/raw_image.jpg
```

Example output:

```text
Input image       : dataset/raw/images/IMG20260704120145.jpg
Undistorted saved : inference/outputs/IMG20260704120145_undistorted.jpg
Annotated saved   : inference/outputs/IMG20260704120145_annotated.jpg
Detections        : 1
  [1] Phone_Cover conf=0.989 bbox=[798, 604, 2046, 2191]
```

### Options

| Flag | Description |
|------|-------------|
| `--output-dir` | Directory for output images (default: `inference/outputs/`) |
| `--weights` | Path to model weights (default: `models/maskrcnn.pth`) |
| `--skip-undistort` | Skip undistortion if input is already corrected |

## Reproducibility

- Dataset split seed: 42 (`dataset/prepare_dataset.py`)
- Training config: `models/config.py`
- Evaluation thresholds: confidence 0.5, mask 0.5
- Model weights stored locally at `models/maskrcnn.pth` (gitignored due to size)

## Limitations

- Small test set (9 images) limits statistical confidence in mAP estimates
- No online augmentation may reduce robustness to unseen lighting or backgrounds
- Training was performed on Google Colab; exact GPU hardware may vary between runs
- RMS calibration error (1.01 px) is above the recommended 0.5 px threshold
