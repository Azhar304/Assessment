"""
Evaluate Mask R-CNN on the held-out test set.

Computes mAP@0.5, mAP@0.5:0.95, mean mask IoU, precision, recall, and F1.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from torchvision.transforms import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_utils import load_model

TEST_DIR = PROJECT_ROOT / "dataset" / "test"
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


def polygon_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


@torch.no_grad()
def collect_predictions(model, device, coco_gt):
    results = []
    matched_ious = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_info in coco_gt.loadImgs(coco_gt.getImgIds()):
        image_path = TEST_DIR / "images" / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tensor = F.to_tensor(image).to(device)
        prediction = model([tensor])[0]

        ann_ids = coco_gt.getAnnIds(imgIds=image_info["id"])
        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_masks = []
        for ann in gt_anns:
            if ann.get("segmentation"):
                gt_masks.append(polygon_to_mask(ann["segmentation"], height, width))

        pred_masks = []
        pred_scores = []
        for score, pred_mask in zip(
            prediction["scores"].cpu().numpy(),
            prediction["masks"].cpu().numpy(),
        ):
            if score < CONFIDENCE_THRESHOLD:
                continue
            pred_masks.append(pred_mask[0] > MASK_THRESHOLD)
            pred_scores.append(float(score))

            rle = mask_utils.encode(
                np.asfortranarray((pred_mask[0] > MASK_THRESHOLD).astype(np.uint8))
            )
            rle["counts"] = rle["counts"].decode("utf-8")
            box = prediction["boxes"][len(pred_scores) - 1].cpu().numpy()
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id": image_info["id"],
                    "category_id": 1,
                    "segmentation": rle,
                    "score": float(score),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                }
            )

        # Greedy IoU matching for summary metrics
        used_preds, used_gts = set(), set()
        pairs = []
        for pi, pred in enumerate(pred_masks):
            for gi, gt in enumerate(gt_masks):
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                iou = float(inter / union) if union else 0.0
                pairs.append((iou, pi, gi))
        pairs.sort(reverse=True)
        for iou, pi, gi in pairs:
            if pi in used_preds or gi in used_gts:
                continue
            used_preds.add(pi)
            used_gts.add(gi)
            matched_ious.append(iou)

        tp = len(used_preds)
        fn = len(gt_masks) - len(used_gts)
        fp = len(pred_masks) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return results, matched_ious, total_tp, total_fp, total_fn


def main():
    print("=" * 60)
    print("Mask R-CNN Test Set Evaluation")
    print("=" * 60)

    annotation_path = TEST_DIR / "annotations.json"
    coco_gt = COCO(str(annotation_path))
    model, device = load_model()

    results, matched_ious, total_tp, total_fp, total_fn = collect_predictions(
        model, device, coco_gt
    )

    if results:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_50 = float(coco_eval.stats[1])
        map_50_95 = float(coco_eval.stats[0])
    else:
        map_50 = 0.0
        map_50_95 = 0.0

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    metrics = {
        "test_images": len(coco_gt.getImgIds()),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "mAP_50": round(map_50, 4),
        "mAP_50_95": round(map_50_95, 4),
        "mean_mask_iou": round(mean_iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }

    output_path = PROJECT_ROOT / "models" / "evaluation_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nTest images       : {metrics['test_images']}")
    print(f"mAP@0.5           : {metrics['mAP_50']:.4f}")
    print(f"mAP@0.5:0.95      : {metrics['mAP_50_95']:.4f}")
    print(f"Mean mask IoU     : {metrics['mean_mask_iou']:.4f}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"Recall            : {metrics['recall']:.4f}")
    print(f"F1                : {metrics['f1']:.4f}")
    print(f"TP / FP / FN      : {total_tp} / {total_fp} / {total_fn}")
    print(f"\nSaved metrics to  : {output_path}")


if __name__ == "__main__":
    main()
