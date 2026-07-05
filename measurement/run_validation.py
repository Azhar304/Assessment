

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from measure_object import measure_image  # noqa: E402
from validate_measurements import compute_validation, load_rows  # noqa: E402


# Mobile phone cover:
GT_WIDTH_MM = 80.0    # short side
GT_HEIGHT_MM = 170.0  # long side

VALIDATION_IMAGES_DIR = BASE_DIR / "validation_images"
OUTPUT_DIR = BASE_DIR / "outputs"
CSV_PATH = BASE_DIR / "validation_results.csv"
SUMMARY_PATH = BASE_DIR / "validation_summary.json"

CSV_FIELDS = ["sample", "image", "gt_width_mm", "gt_height_mm", "pred_width_mm", "pred_height_mm"]


def run_batch():
    image_paths = sorted(VALIDATION_IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {VALIDATION_IMAGES_DIR}")

    rows = []
    failures = []

    print("=" * 60)
    print(f"Phase 3 Validation - {len(image_paths)} image(s)")
    print("=" * 60)

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n--- [{idx}/{len(image_paths)}] {image_path.name} ---")
        try:
            measurement = measure_image(
                image_path=image_path,
                output_dir=OUTPUT_DIR,
            )
        except Exception as exc:
            print(f"[FAILED] {image_path.name}: {exc}")
            failures.append({"image": image_path.name, "error": str(exc)})
            continue

        rows.append(
            {
                "sample": idx,
                "image": image_path.name,
                "gt_width_mm": GT_WIDTH_MM,
                "gt_height_mm": GT_HEIGHT_MM,
                "pred_width_mm": measurement["width_mm"],
                "pred_height_mm": measurement["height_mm"],
            }
        )

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-image predictions to: {CSV_PATH}")

    if failures:
        print(f"\n{len(failures)} image(s) failed detection/inference and were excluded:")
        for failure in failures:
            print(f"  - {failure['image']}: {failure['error']}")
        print(
            "Note these in MEASUREMENT_REPORT.md under Limitations -- "
            "this is expected, honest data, not something to hide."
        )

    if not rows:
        print("\nNo successful measurements; cannot compute validation summary.")
        return

   
    validation_rows = load_rows(CSV_PATH)
    summary = compute_validation(validation_rows)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Samples          : {summary['samples']}")
    print(f"Width MAE        : {summary['width_mae_mm']:.2f} mm")
    print(f"Height MAE       : {summary['height_mae_mm']:.2f} mm")
    print(f"Overall MAE      : {summary['overall_mae_mm']:.2f} mm")
    print(f"Width MPE        : {summary['width_mpe_percent']:.2f}%")
    print(f"Height MPE       : {summary['height_mpe_percent']:.2f}%")
    print(f"Overall MPE      : {summary['overall_mpe_percent']:.2f}%")
    print(f"Saved summary to : {SUMMARY_PATH}")

    print("\nPer-sample breakdown:")
    for row in summary["rows"]:
        print(
            f"  {row['image']:<12} "
            f"W: gt={row['gt_width_mm']:.1f} pred={row['pred_width_mm']:.1f} "
            f"err={row['width_abs_error_mm']:.2f}mm ({row['width_percentage_error']:.1f}%)  |  "
            f"H: gt={row['gt_height_mm']:.1f} pred={row['pred_height_mm']:.1f} "
            f"err={row['height_abs_error_mm']:.2f}mm ({row['height_percentage_error']:.1f}%)"
        )


if __name__ == "__main__":
    run_batch()