# Model Artifacts


The local file `models/maskrcnn.pth` is a trained weight artifact. It is intentionally ignored by Git because it is large. Store and share trained weights through Google Drive.

## Dataset

Training uses the COCO-style phone-cover dataset stored in Google Drive:

```text
/content/drive/MyDrive/dataset/
  train/
    images/
    annotations.json
  val/
    images/
    annotations.json
  test/
    images/
    annotations.json
```

Split counts:

```text
Train: 62 images
Validation: 19 images
Test: 9 images
```

## Training Configuration

```text
Optimizer: SGD
Learning rate: 0.005
Momentum: 0.9
Weight decay: 0.0005
Scheduler: StepLR
Step size: 10 epochs
Gamma: 0.1
Epochs: 15
Train batch size: 2
Validation batch size: 1
Inference confidence threshold: 0.5
Mask threshold: 0.5
```

The same values are recorded in `models/config.py` for reproducibility.

## Current Training Result

The current run completed 15 epochs.

```text
Final train loss: 0.062234348106768825
Final validation loss: 0.08831389111123587
Best validation loss: 0.0873759237951354
Best validation epoch: 13
```

The full loss history is stored in `models/training_history.json`.
