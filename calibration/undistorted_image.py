from pathlib import Path
import cv2
import numpy as np

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "undistorted"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# Load Calibration Parameters
# ==========================================================

camera_matrix = np.load(RESULT_DIR / "camera_matrix.npy")
dist_coeffs = np.load(RESULT_DIR / "dist_coeffs.npy")

# ==========================================================
# Find Images
# ==========================================================

image_paths = sorted(IMAGE_DIR.glob("*"))

supported_ext = {".jpg", ".jpeg", ".png"}

image_paths = [
    img for img in image_paths
    if img.suffix.lower() in supported_ext
]

if len(image_paths) == 0:
    raise FileNotFoundError("No images found.")

print("=" * 60)
print("Image Undistortion")
print("=" * 60)

count = 0

# ==========================================================
# Undistort Images
# ==========================================================

for image_path in image_paths:

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

    count += 1

    print(f"[OK] {image_path.name}")

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("Undistortion Summary")
print("=" * 60)

print(f"Images Processed : {count}")
print(f"Output Folder    : {OUTPUT_DIR}")

print("\nCompleted successfully.")