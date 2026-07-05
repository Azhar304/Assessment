# Camera Calibration Report

## Objective

The objective of calibration is to estimate the camera intrinsic matrix and lens distortion coefficients so every image used for segmentation and measurement can be undistorted before pixel-level processing. This is mandatory for the assessment because raw distorted images do not preserve metric relationships across the image plane.

## Calibration Target

| Parameter | Value |
|-----------|-------|
| Target type | ChArUco board |
| OpenCV dictionary | `DICT_4X4_50` |
| Board size | 7 x 10 squares |
| Square length | 25 mm |
| Marker length | 18 mm |
| Generated board | `calibration/charuco_board.png` |
| Generator script | `calibration/generate_charuco_board.py` |

ChArUco was selected because it combines ArUco marker IDs with chessboard corner interpolation. This is more robust than a plain checkerboard when the board is rotated, partially visible, or captured under varied viewpoints.

## Image Collection

| Item | Value |
|------|-------|
| Image directory | `calibration/images/` |
| Total images | 22 |
| Valid images used | 22 |
| Image resolution | 3120 x 3120 px |
| Capture strategy | Varied angle, distance, scale, and board position |

The assessment requires at least 20 calibration images. This project uses 22 valid calibration images.

## Calibration Method

Implemented in:

```text
calibration/calibrate_camera.py
```

Processing flow:

1. Load all `.jpg` images from `calibration/images/`.
2. Convert each image to grayscale.
3. Detect ArUco markers with OpenCV.
4. Interpolate ChArUco corners from detected markers.
5. Reject images with fewer than 15 interpolated ChArUco corners.
6. Run `cv2.aruco.calibrateCameraCharuco`.
7. Save the camera matrix and distortion coefficients as NumPy arrays.

## Calibration Parameters

Saved outputs:

```text
calibration/results/camera_matrix.npy
calibration/results/dist_coeffs.npy
```

Camera matrix:

```text
[[3011.39568,    0.00000, 1569.67304],
 [   0.00000, 3016.43549, 1580.52300],
 [   0.00000,    0.00000,    1.00000]]
```

Interpretation:

- `fx = 3011.39568`
- `fy = 3016.43549`
- `cx = 1569.67304`
- `cy = 1580.52300`

Distortion coefficients in OpenCV order `[k1, k2, p1, p2, k3]`:

```text
[[ 0.07635520, 0.04910426, 0.00223933, -0.00066068, -0.56525062]]
```

Interpretation:

- `k1`, `k2`, and `k3` model radial distortion.
- `p1` and `p2` model tangential distortion.

## Reprojection Error

| Metric | Value |
|--------|-------|
| RMS reprojection error | 1.008605 px |
| Valid calibration images | 22 / 22 |

The assessment guidance states that values below 0.5 px are acceptable and below 0.3 px are excellent. The current error is above that target (1.01 px). The calibration is still usable for the implemented pipeline, but this is a documented limitation and the most important technical improvement opportunity.

Likely causes:

- Some calibration images may not cover the full image frame evenly.
- Board corner detection may be affected by blur, glare, or board flatness.
- More extreme viewpoints and corner coverage may be needed.

Recommended improvement:

- Recapture 30-40 sharp calibration images.
- Include board positions near all frame corners and edges.
- Avoid motion blur and reflections.
- Re-run `calibration/calibrate_camera.py` and compare RMS error.

## Undistortion Pipeline

Two scripts apply the saved calibration parameters:

| Script | Input | Output |
|--------|-------|--------|
| `calibration/undistorted_image.py` | `calibration/images/` | `calibration/undistorted/` |
| `calibration/undistort_dataset.py` | `dataset/raw/images/` | `dataset/undistorted/images/` |

Both use:

```text
cv2.getOptimalNewCameraMatrix(...)
cv2.undistort(...)
```

The returned ROI is used to crop invalid black borders after undistortion.

## Calibration Dependency

All segmentation and measurement images should be processed with the same calibration parameters. Raw images can produce incorrect metric results because radial and tangential distortion change apparent pixel distances depending on where the object appears in the image.

In this project:

- Dataset images are undistorted before labelling and training.
- Inference images are undistorted by `inference/run_inference.py`.
- Measurement images are undistorted by `measurement/measure_object.py` unless `--skip-undistort` is explicitly provided.

## Verification Checklist

| Requirement | Status |
|-------------|--------|
| Checkerboard/ChArUco calibration target | Met |
| 20+ calibration images | Met, 22 images |
| Intrinsic matrix saved | Met |
| Distortion coefficients saved | Met |
| Reprojection error reported | Met |
| Undistortion applied before measurement | Met |
| Limitation documented | Met, RMS error is above target |
