# Dataset Card

## Object Selection

| Item | Value |
|------|-------|
| Object | Phone cover (silicone/rubber case) |
| Class name | `Phone_Cover` |
| Approximate dimensions | ~160 mm height x ~80 mm width (varies by phone model) |

### Justification

- Readily available and easy to photograph repeatedly
- Flat, rectangular geometry suitable for width/height measurement
- Distinct edges and colour contrast against common backgrounds
- Single connected region — straightforward to label with polygon segmentation

## Collection Strategy

| Item | Value |
|------|-------|
| Camera | Same device used for calibration |
| Raw image directory | `dataset/raw/images/` |
| Total raw images captured | 92 |
| Labelled images | 90 |
| Image resolution | ~2838 x 2879 px (varies slightly after undistortion crop) |

Images were captured with varied:

- Distance from object (close-up to ~50 cm)
- Viewing angle (top-down, oblique, rotated)
- Background surfaces (desk, floor, hand-held)
- Lighting conditions (indoor natural and artificial light)

All raw images were undistorted using `calibration/undistort_dataset.py` before labelling and training.

## Labelling

| Item | Value |
|------|-------|
| Labelling tool | CVAT (Computer Vision Annotation Tool) |
| Annotation type | Instance segmentation (polygon) |
| Export format | COCO 1.0 (`instances_default.json`) |
| Annotations directory | `dataset/undistorted/annotations/` |

Each image contains one `Phone_Cover` instance annotated with a polygon segmentation mask and bounding box.

## Dataset Splits

Created by `dataset/prepare_dataset.py` with random seed 42:

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 62 | 68.9% |
| Validation | 19 | 21.1% |
| Test | 9 | 10.0% |
| **Total** | **90** | **100%** |

Split directories:

```text
dataset/train/images/ + annotations.json
dataset/val/images/   + annotations.json
dataset/test/images/  + annotations.json
```

## Class Distribution

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Phone_Cover | 62 | 19 | 9 | 90 |

Single-class dataset with one instance per image. Class balance is uniform (100% Phone_Cover).

## Augmentation Strategy

No online data augmentation was applied during training. The dataset diversity comes from varied capture conditions (angle, distance, lighting, background). Future work could add random flips, colour jitter, or scale transforms to improve generalisation.

## Dataset Requirements Checklist

| Requirement | Status |
|-------------|--------|
| 70+ object images | Met (90 labelled) |
| Self-collected and self-labelled | Met |
| Undistorted before training | Met |
| Train/val/test split | Met (70/20/10 target) |
| COCO export format | Met |
