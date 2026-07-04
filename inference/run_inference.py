"""
Run Mask R-CNN inference on a single image.

Pipeline:
1. Load raw image
2. Undistort using calibration parameters
3. Run segmentation inference
4. Save annotated output with detected mask overlay
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_utils import load_calibration, load_model, undistort_image

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "inference" / "outputs"
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


def annotate_image(image_bgr, prediction, threshold=CONFIDENCE_THRESHOLD):
    """Draw mask overlay, bounding box, and confidence label."""
    annotated = image_bgr.copy()
    overlay = annotated.copy()
    detections = []

    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    masks = prediction["masks"].detach().cpu().numpy()

    for score, box, mask in zip(scores, boxes, masks):
        if score < threshold:
            continue

        mask_bool = mask[0] > MASK_THRESHOLD
        overlay[mask_bool] = (0, 0, 255)  # red mask in BGR

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Phone_Cover {score:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        detections.append(
            {
                "label": "Phone_Cover",
                "confidence": float(score),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
    return annotated, detections


@torch.no_grad()
def run_inference(image_path, output_dir, weights_path=None, skip_undistort=False):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if skip_undistort:
        processed = raw_image
    else:
        processed = undistort_image(raw_image)

    undistorted_path = output_dir / f"{image_path.stem}_undistorted.jpg"
    cv2.imwrite(str(undistorted_path), processed)

    model, device = load_model(weights_path)
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(Image.fromarray(rgb)).to(device)
    prediction = model([tensor])[0]

    annotated, detections = annotate_image(processed, prediction)
    annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    print("=" * 60)
    print("Inference Complete")
    print("=" * 60)
    print(f"Input image       : {image_path}")
    print(f"Undistorted saved : {undistorted_path}")
    print(f"Annotated saved   : {annotated_path}")
    print(f"Detections        : {len(detections)}")
    for i, det in enumerate(detections, 1):
        print(
            f"  [{i}] {det['label']} "
            f"conf={det['confidence']:.3f} "
            f"bbox={det['bbox']}"
        )

    return {
        "undistorted_path": str(undistorted_path),
        "annotated_path": str(annotated_path),
        "detections": detections,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run undistortion + Mask R-CNN inference on an image."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image (raw or pre-undistorted)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for undistorted and annotated outputs",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: models/maskrcnn.pth)",
    )
    parser.add_argument(
        "--skip-undistort",
        action="store_true",
        help="Skip undistortion if input is already corrected",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        args.image,
        args.output_dir,
        weights_path=args.weights,
        skip_undistort=args.skip_undistort,
    )
