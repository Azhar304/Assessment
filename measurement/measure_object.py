"""
Measure phone-cover width and height in millimetres from a calibrated image.

The script undistorts the input image, runs Mask R-CNN segmentation, extracts the
largest predicted mask contour, and converts pixel dimensions to millimetres using
a pixels_per_mm ratio derived from an automatically detected reference object
(a standard ISO/IEC 7810 ID-1 debit/credit card) in the same undistorted frame.
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

# ISO/IEC 7810 ID-1 standard (debit/credit card) physical dimensions in mm.
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM  # ~1.586
CARD_ASPECT_TOLERANCE = 0.20  # +/-20% tolerance on long/short side ratio (perspective skew)
CARD_MIN_AREA_FRACTION = 0.01   # card must occupy at least 1% of the frame
CARD_MAX_AREA_FRACTION = 0.5    # and no more than 50% of the frame


def _rect_candidates_from_mask(mask, image_area, aspect_ratio, tolerance, min_solidity, source):
    """Find rectangle-like contours in a binary mask that match the target
    aspect ratio and are solidly filled (not just a sparse edge outline)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < CARD_MIN_AREA_FRACTION * image_area:
            continue
        if area > CARD_MAX_AREA_FRACTION * image_area:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (dim_a, dim_b), _ = rect
        short_side, long_side = min(dim_a, dim_b), max(dim_a, dim_b)
        if short_side <= 0:
            continue

        rect_area = dim_a * dim_b
        solidity = area / rect_area if rect_area > 0 else 0
        candidate_ratio = long_side / short_side
        ratio_error = abs(candidate_ratio - aspect_ratio) / aspect_ratio

        if ratio_error <= tolerance and solidity >= min_solidity:
            results.append((area, ratio_error, solidity, rect, source))

    return results


def detect_reference_card(
    image_bgr,
    aspect_ratio=CARD_ASPECT_RATIO,
    tolerance=CARD_ASPECT_TOLERANCE,
    min_solidity=0.80,
    debug_path=None,
):
    
    height, width = image_bgr.shape[:2]
    image_area = height * width

    # --- Strategy A: edge-based ---
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, lower, upper)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    edge_candidates = _rect_candidates_from_mask(
        edges, image_area, aspect_ratio, tolerance, min_solidity, "edges"
    )

    # --- Strategy B: hue-distance background subtraction ---
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    patch = max(20, min(height, width) // 20)
    corners = [
        hsv[0:patch, 0:patch],
        hsv[0:patch, width - patch:width],
        hsv[height - patch:height, 0:patch],
        hsv[height - patch:height, width - patch:width],
    ]
    bg_hue = np.median(np.concatenate([c[..., 0].reshape(-1) for c in corners]))
    hue_diff = np.abs(hsv[..., 0] - bg_hue)
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # hue wraps around at 180
    hue_diff_norm = cv2.normalize(hue_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, hue_mask = cv2.threshold(hue_diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    saturated_enough = (hsv[..., 1] > 25).astype(np.uint8) * 255
    hue_mask = cv2.bitwise_and(hue_mask, saturated_enough)
    hue_mask = cv2.morphologyEx(
        hue_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2
    )
    hue_mask = cv2.morphologyEx(
        hue_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )
    hue_candidates = _rect_candidates_from_mask(
        hue_mask, image_area, aspect_ratio, tolerance, min_solidity, "hue_bg_subtract"
    )

    all_candidates = edge_candidates + hue_candidates

    if debug_path:
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        hue_bgr = cv2.cvtColor(hue_mask, cv2.COLOR_GRAY2BGR)
        annotated = image_bgr.copy()
        for area, ratio_error, solidity, rect, source in all_candidates:
            box = cv2.boxPoints(rect).astype(int)
            cv2.polylines(annotated, [box], True, (0, 255, 0), 3)
            cv2.putText(
                annotated,
                f"{source} area={area:.0f}",
                tuple(box[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        stacked = np.hstack([edge_bgr, hue_bgr, annotated])
        cv2.imwrite(str(debug_path), stacked)

    if not all_candidates:
        return None

    # Rank by area: the true reference card is expected to be the largest
    # solidly-filled, card-ratio-shaped region in frame.
    all_candidates.sort(key=lambda c: -c[0])
    _, _, _, best_rect, best_source = all_candidates[0]
    (center_x, center_y), (dim_a, dim_b), angle = best_rect

    return {
        "center": [float(center_x), float(center_y)],
        "width_px": float(min(dim_a, dim_b)),
        "height_px": float(max(dim_a, dim_b)),
        "angle_degrees": float(angle),
        "box_points": cv2.boxPoints(best_rect).astype(int).tolist(),
        "method": f"auto_detected_card_contour ({best_source})",
    }


def compute_pixels_per_mm(reference_width_px, reference_height_px, reference_width_mm, reference_height_mm):
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


def draw_measurement(image_bgr, mask, measurement, reference_rect, confidence):
    annotated = image_bgr.copy()
    overlay = annotated.copy()
    overlay[mask] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    box_points = np.array(measurement["box_points"], dtype=np.int32)
    cv2.polylines(annotated, [box_points], True, (0, 255, 0), 2)

    ref_points = np.array(reference_rect["box_points"], dtype=np.int32)
    cv2.polylines(annotated, [ref_points], True, (255, 0, 0), 2)
    x, y = ref_points[np.argmin(ref_points.sum(axis=1))]

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
        f"Reference ({reference_rect['method']})",
        (int(x), max(int(y) - 10, 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        3,
    )

    return annotated


@torch.no_grad()
def measure_image(
    image_path,
    reference_width_mm=None,
    reference_height_mm=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    weights_path=None,
    skip_undistort=False,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    debug=False,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    processed = raw_image if skip_undistort else undistort_image(raw_image)

    debug_path = (
        output_dir / f"{image_path.stem}_card_detection_debug.jpg"
        if debug
        else None
    )
    reference_rect = detect_reference_card(processed, debug_path=debug_path)
    if reference_rect is None:
        hint = (
            f" Debug image saved to {debug_path}."
            if debug
            else " Re-run with --debug to save a diagnostic image showing every candidate contour."
        )
        raise RuntimeError(
            "Could not auto-detect a card-shaped reference object in this "
            "image. Improve lighting/contrast between the card and the "
            "background, or make sure the whole card is visible and not "
            "occluded." + hint
        )

    # Default to the standard ISO/IEC 7810 ID-1 card size unless overridden.
    if reference_width_mm is None and reference_height_mm is None:
        reference_width_mm = CARD_WIDTH_MM
        reference_height_mm = CARD_HEIGHT_MM

    pixels_per_mm = compute_pixels_per_mm(
        reference_rect["width_px"],
        reference_rect["height_px"],
        reference_width_mm,
        reference_height_mm,
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
            "reference_method": reference_rect["method"],
            "reference_box_points": reference_rect["box_points"],
            "reference_width_px": reference_rect["width_px"],
            "reference_height_px": reference_rect["height_px"],
            "reference_width_mm": reference_width_mm,
            "reference_height_mm": reference_height_mm,
            "undistortion_applied": not skip_undistort,
        }
    )

    annotated = draw_measurement(processed, mask, measurement, reference_rect, confidence)
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
    print(f"Reference method  : {reference_rect['method']}")
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
        "--reference-width-mm",
        type=float,
        default=None,
        help=(
            "Real reference width in millimetres (default: ISO/IEC 7810 "
            "ID-1 card width, 85.60mm)"
        ),
    )
    parser.add_argument(
        "--reference-height-mm",
        type=float,
        default=None,
        help=(
            "Real reference height in millimetres (default: ISO/IEC 7810 "
            "ID-1 card height, 53.98mm)"
        ),
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save a diagnostic image showing every candidate contour "
            "considered during auto reference-card detection (green = "
            "passed filters, red = rejected), alongside the edge map."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    measure_image(
        image_path=args.image,
        reference_width_mm=args.reference_width_mm,
        reference_height_mm=args.reference_height_mm,
        output_dir=args.output_dir,
        weights_path=args.weights,
        skip_undistort=args.skip_undistort,
        confidence_threshold=args.confidence_threshold,
        debug=args.debug,
    )