# Measurement Report

## Status

Step 3 pixel-to-mm measurement is implemented in `measurement/`.

Physical accuracy validation with 10+ ruler/calliper samples is still pending.

## Methodology

The end-to-end measurement pipeline:

1. Accepts a raw image from the calibrated camera
2. Undistorts the image using intrinsic parameters from Step 1
3. Uses a known-size reference object bbox to compute a `pixels_per_mm` ratio
4. Runs Mask R-CNN inference to segment the phone cover
5. Extracts pixel width and height from the segmentation mask contour
6. Converts pixel dimensions to millimetres
7. Outputs an annotated image with width (mm), height (mm), and confidence score

## Implementation

| File | Purpose |
|------|---------|
| `measurement/measure_object.py` | End-to-end measurement for one image |
| `measurement/validate_measurements.py` | MAE/MPE validation from CSV |
| `measurement/validation_template.csv` | 10-sample validation template |

Example usage:

```powershell
python measurement/measure_object.py path/to/image.jpg --reference-bbox x,y,width,height --reference-width-mm 85.6 --reference-height-mm 54.0
```

The reference bbox must be measured in the processed image coordinate space. If the input is raw, the script undistorts first and the reference bbox should correspond to the undistorted view. If the input is already undistorted, use `--skip-undistort`.

## Pixel-to-MM Conversion

The conversion ratio is computed from the known reference object:

```text
pixels_per_mm = reference_size_pixels / reference_size_mm
```

When both reference width and height are supplied, the script averages the two ratios for a single conversion factor. The phone cover is measured from the minimum-area rectangle around the predicted mask contour:

```text
object_width_mm = object_width_pixels / pixels_per_mm
object_height_mm = object_height_pixels / pixels_per_mm
```

## Calibration Dependency

All measurement images must be undistorted before pixel-to-mm conversion. The measurement module reuses the same calibration parameters:

- `calibration/results/camera_matrix.npy`
- `calibration/results/dist_coeffs.npy`

Raw images produce incorrect measurements because lens distortion changes apparent pixel dimensions depending on position in the frame.

## Accuracy Validation

The assessment requires 10+ physical validation samples. Fill `measurement/validation_template.csv` with ruler/calliper ground truth and predicted measurements, then run:

```powershell
python measurement/validate_measurements.py measurement/validation_template.csv
```

The script reports:

- Width MAE
- Height MAE
- Overall MAE
- Width MPE
- Height MPE
- Overall MPE

## Current Limitation

The measurement code is implemented, but final accuracy numbers cannot be reported until 10+ real samples are measured physically and added to the validation CSV.
