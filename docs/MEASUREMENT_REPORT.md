# Measurement Report

## Status

**Step 3 (Pixel-to-MM Measurement) is not yet implemented.**

This document will be completed when the measurement pipeline is built in `measurement/`.

## Planned Methodology

Per the assessment requirements, the end-to-end measurement pipeline will:

1. Accept a raw image from the calibrated camera
2. Undistort the image using intrinsic parameters from Step 1
3. Detect a reference object and compute a `pixels_per_mm` ratio
4. Run Mask R-CNN inference to segment the phone cover
5. Extract pixel width and height from the segmentation mask contour
6. Convert pixel dimensions to millimetres
7. Output annotated image with width (mm), height (mm), and confidence score

## Calibration Dependency

All measurement images must be undistorted before pixel-to-mm conversion. The inference script already applies undistortion (`inference/run_inference.py`). The measurement module will reuse the same calibration parameters:

- `calibration/results/camera_matrix.npy`
- `calibration/results/dist_coeffs.npy`

Raw (distorted) images produce incorrect measurements because lens distortion changes apparent pixel dimensions depending on position in the frame.

## Planned Accuracy Validation

When implemented, the measurement module will:

- Measure 10+ object instances with a physical ruler or calliper
- Compare ground-truth vs system output
- Report mean absolute error (MAE) and mean percentage error (MPE)
- Include an error table in this report

## Planned End-to-End Demo

A single script or notebook will take one new image as input and output:

- Segmentation mask overlay
- Width (mm)
- Height (mm)
- Confidence score
