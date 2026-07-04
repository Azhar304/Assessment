"""
Measure phone-cover width and height in millimetres from a calibrated image.

The script undistorts the input image, runs Mask R-CNN segmentation, extracts the
largest predicted mask contour, and converts pixel dimensions to millimetres
using a known reference object's bounding box.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_utils import load_model, undistort_image

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "measurement" / "outputs"
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


def parse_reference_bbox(value):
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Reference bbox must be formatted as x,y,width,height"
        )
    x, y, width, height = parts
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Reference bbox width/height must be > 0")
    return x, y, width, height


def compute_pixels_per_mm(reference_bbox, reference_width_mm, reference_height_mm):
    _, _, reference_width_px, reference_height_px = reference_bbox

    ratios = []
    if reference_width_mm is not None:
        ratios.append(reference_width_px / reference_width_mm)
    if reference_height_mm is not None:
        ratios.append(reference_height_px / reference_height_mm)

    if not ratios:
        raise ValueError("Provide --reference-width-mm, --reference-height-mm, or both")

    return float(np.mean(ratios))


def get_best_prediction(prediction, confidence_threshold):
    scores = prediction["scores"].detach().cpu().numpy()
    valid_indices = np.where(scores >= confidence_threshold)[0]
    if len(valid_indices) == 0:
        return None

    best_idx = int(valid_indices[np.argmax(scores[valid_indices])])
    mask = prediction["masks"][best_idx, 0].detach().cpu().numpy() > MASK_THRESHOLD
    box = prediction["boxes"][best_idx].detach().cpu().numpy()
    score = float(scores[best_idx])
    return mask, box, score


def measure_mask(mask, pixels_per_mm):
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in predicted mask")

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (dim_a, dim_b), angle = rect

    width_px = float(min(dim_a, dim_b))
    height_px = float(max(dim_a, dim_b))

    return {
        "width_px": width_px,
        "height_px": height_px,
        "width_mm": width_px / pixels_per_mm,
        "height_mm": height_px / pixels_per_mm,
        "pixels_per_mm": pixels_per_mm,
        "angle_degrees": float(angle),
        "center": [float(center_x), float(center_y)],
        "contour_area_px": float(cv2.contourArea(contour)),
        "box_points": cv2.boxPoints(rect).astype(int).tolist(),
    }


def draw_measurement(image_bgr, mask, measurement, reference_bbox, confidence):
    annotated = image_bgr.copy()
    overlay = annotated.copy()
    overlay[mask] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    box_points = np.array(measurement["box_points"], dtype=np.int32)
    cv2.polylines(annotated, [box_points], True, (0, 255, 0), 2)

    x, y, width, height = [int(round(v)) for v in reference_bbox]
    cv2.rectangle(annotated, (x, y), (x + width, y + height), (255, 0, 0), 2)

    label_lines = [
        f"Phone_Cover confidence: {confidence:.2f}",
        f"WIDTH:  {measurement['width_mm']:.1f} mm",
        f"LENGTH: {measurement['height_mm']:.1f} mm",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.05
    thickness = 3
    line_height = 42
    padding = 14
    text_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0]
        for line in label_lines
    ]
    panel_width = max(width for width, _ in text_sizes) + padding * 2
    panel_height = line_height * len(label_lines) + padding * 2
    panel_x, panel_y = 16, 16

    cv2.rectangle(
        annotated,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        annotated,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (0, 255, 255),
        2,
    )

    for idx, line in enumerate(label_lines):
        baseline_y = panel_y + padding + 32 + idx * line_height
        cv2.putText(
            annotated,
            line,
            (panel_x + padding, baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (panel_x + padding, baseline_y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        "Reference",
        (x, max(y - 10, 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        3,
    )

    return annotated


@torch.no_grad()
def measure_image(
    image_path,
    reference_bbox,
    reference_width_mm=None,
    reference_height_mm=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    weights_path=None,
    skip_undistort=False,
    confidence_threshold=CONFIDENCE_THRESHOLD,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    processed = raw_image if skip_undistort else undistort_image(raw_image)
    pixels_per_mm = compute_pixels_per_mm(
        reference_bbox, reference_width_mm, reference_height_mm
    )

    model, device = load_model(weights_path)
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(Image.fromarray(rgb)).to(device)
    prediction = model([tensor])[0]

    best_prediction = get_best_prediction(prediction, confidence_threshold)
    if best_prediction is None:
        raise RuntimeError("No prediction found above confidence threshold")

    mask, box, confidence = best_prediction
    measurement = measure_mask(mask, pixels_per_mm)
    measurement.update(
        {
            "image": str(image_path),
            "confidence": confidence,
            "bbox_xyxy": [float(v) for v in box],
            "reference_bbox_xywh": [float(v) for v in reference_bbox],
            "reference_width_mm": reference_width_mm,
            "reference_height_mm": reference_height_mm,
            "undistortion_applied": not skip_undistort,
        }
    )

    annotated = draw_measurement(processed, mask, measurement, reference_bbox, confidence)
    annotated_path = output_dir / f"{image_path.stem}_measurement.jpg"
    json_path = output_dir / f"{image_path.stem}_measurement.json"

    cv2.imwrite(str(annotated_path), annotated)
    with open(json_path, "w") as f:
        json.dump(measurement, f, indent=4)

    print("=" * 60)
    print("Measurement Complete")
    print("=" * 60)
    print(f"Input image       : {image_path}")
    print(f"Annotated output  : {annotated_path}")
    print(f"JSON output       : {json_path}")
    print(f"Confidence        : {confidence:.3f}")
    print(f"Pixels per mm     : {pixels_per_mm:.4f}")
    print(f"Width             : {measurement['width_mm']:.2f} mm")
    print(f"Height            : {measurement['height_mm']:.2f} mm")

    return measurement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure phone-cover dimensions from segmentation mask."
    )
    parser.add_argument("image", type=str, help="Path to raw or undistorted input image")
    parser.add_argument(
        "--reference-bbox",
        type=parse_reference_bbox,
        required=True,
        help="Known reference bbox in processed image coordinates: x,y,width,height",
    )
    parser.add_argument(
        "--reference-width-mm",
        type=float,
        default=None,
        help="Real reference width in millimetres",
    )
    parser.add_argument(
        "--reference-height-mm",
        type=float,
        default=None,
        help="Real reference height in millimetres",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for measurement outputs",
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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Minimum model confidence for measurement",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    measure_image(
        image_path=args.image,
        reference_bbox=args.reference_bbox,
        reference_width_mm=args.reference_width_mm,
        reference_height_mm=args.reference_height_mm,
        output_dir=args.output_dir,
        weights_path=args.weights,
        skip_undistort=args.skip_undistort,
        confidence_threshold=args.confidence_threshold,
    )
