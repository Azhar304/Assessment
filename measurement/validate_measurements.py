"""
Compute measurement validation error from a CSV file.

Expected CSV columns:
sample,image,gt_width_mm,gt_height_mm,pred_width_mm,pred_height_mm
"""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "validation_summary.json"


def to_float(row, key):
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"Missing value for {key}")
    return float(value)


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def compute_validation(rows):
    if not rows:
        raise ValueError("Validation CSV has no rows")

    evaluated = []
    width_errors = []
    height_errors = []
    width_percentage_errors = []
    height_percentage_errors = []

    for row in rows:
        gt_width = to_float(row, "gt_width_mm")
        gt_height = to_float(row, "gt_height_mm")
        pred_width = to_float(row, "pred_width_mm")
        pred_height = to_float(row, "pred_height_mm")

        width_error = abs(pred_width - gt_width)
        height_error = abs(pred_height - gt_height)
        width_percentage = (width_error / gt_width) * 100 if gt_width else 0.0
        height_percentage = (height_error / gt_height) * 100 if gt_height else 0.0

        width_errors.append(width_error)
        height_errors.append(height_error)
        width_percentage_errors.append(width_percentage)
        height_percentage_errors.append(height_percentage)

        evaluated.append(
            {
                "sample": row.get("sample", ""),
                "image": row.get("image", ""),
                "gt_width_mm": gt_width,
                "gt_height_mm": gt_height,
                "pred_width_mm": pred_width,
                "pred_height_mm": pred_height,
                "width_abs_error_mm": width_error,
                "height_abs_error_mm": height_error,
                "width_percentage_error": width_percentage,
                "height_percentage_error": height_percentage,
            }
        )

    mean_width_error = sum(width_errors) / len(width_errors)
    mean_height_error = sum(height_errors) / len(height_errors)
    all_abs_errors = width_errors + height_errors
    all_percentage_errors = width_percentage_errors + height_percentage_errors

    return {
        "samples": len(rows),
        "width_mae_mm": mean_width_error,
        "height_mae_mm": mean_height_error,
        "overall_mae_mm": sum(all_abs_errors) / len(all_abs_errors),
        "width_mpe_percent": sum(width_percentage_errors) / len(width_percentage_errors),
        "height_mpe_percent": sum(height_percentage_errors) / len(height_percentage_errors),
        "overall_mpe_percent": sum(all_percentage_errors) / len(all_percentage_errors),
        "rows": evaluated,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute MAE and MPE for physical measurement validation."
    )
    parser.add_argument("csv", type=str, help="Validation CSV path")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    summary = compute_validation(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 60)
    print("Measurement Validation")
    print("=" * 60)
    print(f"Samples          : {summary['samples']}")
    print(f"Width MAE        : {summary['width_mae_mm']:.2f} mm")
    print(f"Height MAE       : {summary['height_mae_mm']:.2f} mm")
    print(f"Overall MAE      : {summary['overall_mae_mm']:.2f} mm")
    print(f"Overall MPE      : {summary['overall_mpe_percent']:.2f}%")
    print(f"Saved summary to : {output_path}")


if __name__ == "__main__":
    main()
