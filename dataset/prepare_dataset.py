from pathlib import Path
import json
import random
import shutil

# Configuration

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "dataset" / "undistorted" / "images"
ANNOTATION_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "undistorted"
    / "annotations"
    / "instances_default.json"
)

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
VAL_DIR = PROJECT_ROOT / "dataset" / "val"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"

# Create folders

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    (folder / "images").mkdir(parents=True, exist_ok=True)

# Load COCO

with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

random.shuffle(images)

n = len(images)

train_end = int(0.70 * n)
val_end = int(0.90 * n)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]


def save_split(split_images, split_dir, json_name="annotations.json"):
    image_output_dir = split_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    for existing_image in image_output_dir.iterdir():
        if existing_image.is_file():
            existing_image.unlink()

    image_ids = {img["id"] for img in split_images}

    split_annotations = [
        ann
        for ann in annotations
        if ann["image_id"] in image_ids
    ]

    split_json = {
        "images": split_images,
        "annotations": split_annotations,
        "categories": categories,
    }

    with open(split_dir / json_name, "w") as f:
        json.dump(split_json, f, indent=4)

    for img in split_images:
        src = IMAGE_DIR / img["file_name"]
        dst = image_output_dir / img["file_name"]

        if src.exists():
            shutil.copy2(src, dst)


save_split(train_images, TRAIN_DIR)
save_split(val_images, VAL_DIR)
save_split(test_images, TEST_DIR)

print("=" * 60)
print("Dataset prepared successfully")
print("=" * 60)

print(f"Train : {len(train_images)} images")
print(f"Val   : {len(val_images)} images")
print(f"Test  : {len(test_images)} images")
