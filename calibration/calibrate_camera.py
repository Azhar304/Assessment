from pathlib import Path
import cv2
import numpy as np

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)

# ==========================================================
# ChArUco Board Configuration
# ==========================================================

# MUST match generate_charuco_board.py
SQUARES_X = 7
SQUARES_Y = 10

SQUARE_LENGTH = 0.025      # 25 mm
MARKER_LENGTH = 0.018      # 18 mm

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

detector = cv2.aruco.ArucoDetector(
    dictionary,
    cv2.aruco.DetectorParameters()
)

# ==========================================================
# Storage
# ==========================================================

all_charuco_corners = []
all_charuco_ids = []

image_size = None

total_images = 0
valid_images = 0

print("=" * 60)
print("Intrinsic Camera Calibration")
print("=" * 60)

image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

if len(image_paths) == 0:
    raise FileNotFoundError("No calibration images found.")

# ==========================================================
# Detect ChArUco Corners
# ==========================================================

for image_path in image_paths:

    total_images += 1

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[FAILED] {image_path.name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        print(f"[FAILED] {image_path.name} -> No markers detected")
        continue

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )

    if retval < 15:
        print(f"[REJECTED] {image_path.name} -> {retval} corners")
        continue

    valid_images += 1

    all_charuco_corners.append(charuco_corners)
    all_charuco_ids.append(charuco_ids)

    if image_size is None:
        image_size = gray.shape[::-1]

    print(
        f"[OK] {image_path.name:<8} "
        f"Markers: {len(ids):2d} "
        f"Corners: {retval:2d}"
    )

# ==========================================================
# Calibration
# ==========================================================

print("\nRunning camera calibration...\n")

rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
    cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

# ==========================================================
# Save Results
# ==========================================================

np.save(
    RESULT_DIR / "camera_matrix.npy",
    camera_matrix
)

np.save(
    RESULT_DIR / "dist_coeffs.npy",
    dist_coeffs
)

# ==========================================================
# Summary
# ==========================================================

print("=" * 60)
print("Calibration Summary")
print("=" * 60)

print(f"Total Images : {total_images}")
print(f"Valid Images : {valid_images}")
print(f"Rejected     : {total_images - valid_images}")

print(f"\nImage Size   : {image_size[0]} x {image_size[1]}")

print(f"\nRMS Error    : {rms:.6f}")

print("\nCamera Matrix\n")
print(camera_matrix)

print("\nDistortion Coefficients\n")
print(dist_coeffs)

print("\nSaved Files")
print("----------------------------")
print("camera_matrix.npy")
print("dist_coeffs.npy")

print("\nCalibration completed successfully.")