# Camera Calibration Report

## Objective

Estimate intrinsic camera parameters and lens distortion coefficients so images can be undistorted before segmentation and metric measurement. Pixel measurements from distorted images are not geometrically reliable.

## Calibration Target

| Parameter | Value |
|-----------|-------|
| Target type | ChArUco board |
| Dictionary | OpenCV `DICT_4X4_50` |
| Board size | 7 x 10 squares |
| Square length | 25 mm |
| Marker length | 18 mm |
| Board image | `calibration/charuco_board.png` |
| Generator script | `calibration/generate_charuco_board.py` |

ChArUco combines ArUco marker detection with chessboard corner interpolation, improving robustness when the board is partially visible or captured from varied angles.

## Image Collection

| Item | Value |
|------|-------|
| Image directory | `calibration/images/` |
| Total images collected | 22 |
| Image resolution | 3120 x 3120 px |
| Capture strategy | Varied board positions, angles, and distances |

The assessment requires at least 20 calibration images. The current dataset satisfies that requirement.

## Calibration Method

Implemented in `calibration/calibrate_camera.py`.

Processing steps:

1. Load all `.jpg` images from `calibration/images/`.
2. Convert each image to grayscale.
3. Detect ArUco markers.
4. Interpolate ChArUco corners from detected markers.
5. Reject images with fewer than 15 interpolated ChArUco corners.
6. Run `cv2.aruco.calibrateCameraCharuco`.
7. Save intrinsics and distortion coefficients as NumPy arrays.

## Calibration Outputs

Saved files:

- `calibration/results/camera_matrix.npy`
- `calibration/results/dist_coeffs.npy`

Camera matrix:

```text
[[3011.39568,    0.00000, 1569.67304],
 [   0.00000, 3016.43549, 1580.52300],
 [   0.00000,    0.00000,    1.00000]]
```

Distortion coefficients (OpenCV order `[k1, k2, p1, p2, k3]`):

```text
[[ 0.07635520, 0.04910426, 0.00223933, -0.00066068, -0.56525062]]
```

## Reprojection Error

| Metric | Value |
|--------|-------|
| RMS reprojection error | 1.008605 px |
| Valid calibration images | 22 / 22 |

The assessment guidance states values below 0.5 px are acceptable and below 0.3 px is excellent. The current RMS error is above that threshold and may be improved with additional calibration images, tighter corner detection, or recapture at more varied angles.

## Undistortion Pipeline

Two undistortion scripts apply the saved calibration parameters:

| Script | Input | Output |
|--------|-------|--------|
| `calibration/undistorted_image.py` | `calibration/images/` | `calibration/undistorted/` |
| `calibration/undistort_dataset.py` | `dataset/raw/images/` | `dataset/undistorted/images/` |

Both use `cv2.getOptimalNewCameraMatrix` followed by `cv2.undistort`, with ROI cropping to remove black borders.

All 22 calibration images and 92 raw dataset images have been undistorted and verified.

## Calibration Dependency for Measurement

All images used for real-world measurement must be undistorted before pixel-to-mm conversion. Raw images contain radial and tangential distortion that changes apparent pixel dimensions depending on position in the frame. The inference pipeline (`inference/run_inference.py`) applies undistortion automatically before model inference.
