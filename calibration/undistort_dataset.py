from pathlib import Path
import cv2
import numpy as np


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "dataset" / "raw" / "images"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "undistorted" / "images"

CALIB_DIR = PROJECT_ROOT / "calibration" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load calibration parameters

camera_matrix = np.load(CALIB_DIR / "camera_matrix.npy")
dist_coeffs = np.load(CALIB_DIR / "dist_coeffs.npy")


# Supported image types

extensions = {".jpg", ".jpeg", ".png"}

images = sorted([
    img for img in INPUT_DIR.iterdir()
    if img.suffix.lower() in extensions
])

print("=" * 60)
print("Undistorting Dataset Images")
print("=" * 60)

processed = 0


# Process images

for image_path in images:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[FAILED] {image_path.name}")
        continue

    h, w = image.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        1,
        (w, h)
    )

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    x, y, rw, rh = roi

    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), undistorted)

    processed += 1

    print(f"[OK] {image_path.name}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Images processed : {processed}")
print(f"Output folder    : {OUTPUT_DIR}")