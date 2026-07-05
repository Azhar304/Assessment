# Measurement Report

## Objective

Compute real-world phone-cover width and height in millimetres from calibrated image data. The pipeline starts from a raw image, applies camera undistortion, segments the phone cover, estimates pixels-per-mm from a known reference object, and outputs metric dimensions with an annotated image.

## Status

Step 3 is implemented and validated on 10 measurement images.

| Item | Status |
|------|--------|
| Raw image to measurement pipeline | Complete |
| Undistortion dependency | Complete |
| Reference-based pixels-per-mm conversion | Complete |
| Width and height output | Complete |
| Confidence score output | Complete |
| Annotated mask overlay | Complete |
| 10-sample physical validation | Complete |
| MAE and MPE report | Complete |

## End-to-End Pipeline

```text
raw measurement image
  -> cv2.undistort using Step 1 calibration
  -> automatic ID-1 card reference detection
  -> pixels_per_mm calculation
  -> Mask R-CNN phone-cover segmentation
  -> mask contour extraction
  -> minimum-area rectangle around mask
  -> width_px and height_px
  -> width_mm and height_mm
  -> annotated image + JSON result
```

## Implementation Files

| File | Purpose |
|------|---------|
| `measurement/measure_object.py` | Single-image end-to-end measurement. |
| `measurement/run_validation.py` | Batch runner for validation images. |
| `measurement/validate_measurements.py` | MAE/MPE calculation from CSV. |
| `measurement/validation_results.csv` | Ground truth and predicted measurements for 10 samples. |
| `measurement/validation_summary.json` | Detailed validation summary and per-sample errors. |
| `measurement/outputs/` | Annotated measurement images and JSON outputs. |

## Usage

Measure one image containing the phone cover and a visible card-sized reference object:

```powershell
python measurement/measure_object.py measurement/validation_images/01.jpg
```

Use custom reference dimensions if the reference object is not a standard ID-1 card:

```powershell
python measurement/measure_object.py path/to/image.jpg --reference-width-mm 85.60 --reference-height-mm 53.98
```

Use an already undistorted image:

```powershell
python measurement/measure_object.py path/to/undistorted_image.jpg --skip-undistort
```

Save reference-card diagnostics:

```powershell
python measurement/measure_object.py path/to/image.jpg --debug
```

Validate saved predictions:

```powershell
python measurement/validate_measurements.py measurement/validation_results.csv
```

Run all validation images:

```powershell
python measurement/run_validation.py
```

## Reference Object

The measurement script automatically detects a card-shaped reference object using the ISO/IEC 7810 ID-1 dimensions:

| Reference dimension | Value |
|---------------------|-------|
| Width | 85.60 mm |
| Height | 53.98 mm |
| Aspect ratio | 1.586 |

The detector searches for solid rectangular contours with a compatible long-side/short-side aspect ratio. It combines edge-based contour detection with hue-distance background subtraction, then selects the largest plausible card-like region.

## Pixel-to-MM Derivation

The reference card provides the conversion ratio from pixels to millimetres.

For the detected reference card:

```text
width_ratio_px_per_mm  = reference_width_px  / reference_width_mm
height_ratio_px_per_mm = reference_height_px / reference_height_mm
pixels_per_mm          = mean(width_ratio_px_per_mm, height_ratio_px_per_mm)
```

For the phone-cover mask:

```text
object_width_mm  = object_width_px  / pixels_per_mm
object_height_mm = object_height_px / pixels_per_mm
```

The object pixel dimensions are computed from `cv2.minAreaRect` around the largest predicted mask contour. This handles rotated phone covers better than an axis-aligned bounding box.

## Calibration Dependency

All measurement images must be undistorted before pixel-to-mm conversion. This project uses:

```text
calibration/results/camera_matrix.npy
calibration/results/dist_coeffs.npy
```

Raw distorted images produce incorrect measurements because radial and tangential lens distortion change apparent pixel distances across the frame. A phone cover near an image edge may appear stretched or compressed compared with the same object near the centre. Undistortion reduces this position-dependent geometric error before the pixels-per-mm ratio is applied.

## Output Format

Each measurement run writes:

```text
measurement/outputs/<name>_measurement.jpg
measurement/outputs/<name>_measurement.json
```

The JSON contains:

- Source image path.
- Width and height in pixels.
- Width and height in millimetres.
- Pixels-per-mm ratio.
- Model confidence.
- Predicted bounding box.
- Reference-card dimensions in pixels and millimetres.
- Reference detection method.
- Reference contour points.
- Undistortion status.

## Accuracy Validation

Validation was performed on 10 images in:

```text
measurement/validation_images/
```

Ground-truth dimensions:

| Dimension | Value |
|-----------|-------|
| Width | 80.0 mm |
| Height | 170.0 mm |

The validation CSV is:

```text
measurement/validation_results.csv
```

### Per-Sample Results

| Sample | Image | GT Width | Pred Width | Width Error | GT Height | Pred Height | Height Error |
|--------|-------|----------|------------|-------------|-----------|-------------|--------------|
| 1 | `01.jpg` | 80.0 | 82.32 | 2.32 | 170.0 | 171.49 | 1.49 |
| 2 | `02.jpg` | 80.0 | 82.60 | 2.60 | 170.0 | 175.02 | 5.02 |
| 3 | `03.jpg` | 80.0 | 82.32 | 2.32 | 170.0 | 177.79 | 7.79 |
| 4 | `04.jpg` | 80.0 | 82.92 | 2.92 | 170.0 | 159.95 | 10.05 |
| 5 | `05.jpg` | 80.0 | 78.06 | 1.94 | 170.0 | 163.63 | 6.37 |
| 6 | `06.jpg` | 80.0 | 82.36 | 2.36 | 170.0 | 172.65 | 2.65 |
| 7 | `07.jpg` | 80.0 | 79.38 | 0.62 | 170.0 | 170.98 | 0.98 |
| 8 | `08.jpg` | 80.0 | 85.91 | 5.91 | 170.0 | 158.38 | 11.62 |
| 9 | `09.jpg` | 80.0 | 81.99 | 1.99 | 170.0 | 165.31 | 4.69 |
| 10 | `10.jpg` | 80.0 | 83.71 | 3.71 | 170.0 | 171.56 | 1.56 |

All dimensions are in millimetres.

### Summary Metrics

| Metric | Value |
|--------|-------|
| Samples | 10 |
| Width MAE | 2.67 mm |
| Height MAE | 5.22 mm |
| Overall MAE | 3.95 mm |
| Width MPE | 3.34% |
| Height MPE | 3.07% |
| Overall MPE | 3.20% |

## Error Analysis

The overall measurement error is 3.95 mm MAE and 3.20% MPE. Width error is lower than height error. Height error is more sensitive because the phone cover is longer, and small perspective or contour errors accumulate over the long dimension.

Largest observed errors:

- Sample 8 height error: 11.62 mm.
- Sample 4 height error: 10.05 mm.
- Sample 8 width error: 5.91 mm.

Likely error sources:

- Calibration RMS error is above the recommended threshold.
- Phone cover and reference card may not be perfectly coplanar.
- Perspective tilt can change apparent dimensions.
- Automatic reference detection may use a slightly imperfect contour.
- Segmentation masks may include or miss thin boundary regions.

## Design Decisions

- A card-sized reference object is detected automatically to make the demo more end-to-end.
- A known physical reference in the same image is used instead of camera-distance estimation because it is simpler and more reliable for a planar measurement setup.
- The predicted mask contour is used instead of only the model bounding box because masks better follow object boundaries.
- Minimum-area rectangles are used so rotated objects can still be measured.
- Results are saved as both images and JSON for visual inspection and quantitative validation.

## Assumptions

- The phone cover and reference card are in approximately the same physical plane.
- The full reference card is visible.
- The reference object has the expected ID-1 card dimensions unless custom dimensions are supplied.
- The target object is the most confident predicted `Phone_Cover` instance.
- Measurement images are captured with the calibrated camera.

## Limitations

- Strong perspective tilt is not fully corrected by the current 2D pixel-ratio method.
- The automatic card detector depends on contrast, visibility, and card-like geometry.
- Calibration reprojection error should be improved for stricter industrial measurement accuracy.
- Validation uses 10 samples, satisfying the assessment minimum but still limited statistically.

## Requirements Checklist

| Requirement | Status |
|-------------|--------|
| Pixel-to-mm conversion derived | Met |
| Width and height in mm calculated | Met |
| Undistortion before measurement | Met |
| Raw distortion issue documented | Met |
| 10+ physical validation samples | Met |
| MAE reported | Met |
| MPE reported | Met |
| End-to-end script provided | Met |
| Mask overlay output | Met |
| Confidence score output | Met |
| Limitations discussed | Met |
