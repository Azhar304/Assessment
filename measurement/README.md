# Measurement Pipeline

This folder contains the Step 3 pixel-to-mm measurement pipeline. It converts a segmented phone-cover mask into real-world width and height measurements using camera undistortion and a known-size reference card in the same image.

## Method

```text
raw image
  -> camera undistortion
  -> automatic ID-1 card reference detection
  -> pixels-per-mm conversion
  -> Mask R-CNN segmentation
  -> contour extraction from predicted mask
  -> minimum-area rectangle measurement
  -> width/height in millimetres
```

The default reference is an ISO/IEC 7810 ID-1 card:

```text
Width:  85.60 mm
Height: 53.98 mm
```

## Files

| File | Purpose |
|------|---------|
| `measure_object.py` | Measure one image and write annotated output + JSON. |
| `run_validation.py` | Run all images in `validation_images/` and produce validation CSV/summary. |
| `validate_measurements.py` | Compute MAE and MPE from a validation CSV. |
| `validation_results.csv` | 10-sample ground truth and prediction table. |
| `validation_summary.json` | Detailed validation metrics and per-sample errors. |
| `validation_images/` | Images used for physical validation. |
| `outputs/` | Annotated measurement images and JSON outputs. |

## Measure One Image

```powershell
python measurement/measure_object.py measurement/validation_images/01.jpg
```

Optional flags:

```powershell
python measurement/measure_object.py path/to/image.jpg --reference-width-mm 85.60 --reference-height-mm 53.98
python measurement/measure_object.py path/to/image.jpg --weights models/maskrcnn.pth
python measurement/measure_object.py path/to/undistorted_image.jpg --skip-undistort
python measurement/measure_object.py path/to/image.jpg --debug
```

Outputs:

```text
measurement/outputs/<image>_measurement.jpg
measurement/outputs/<image>_measurement.json
```

## Validate Accuracy

Run the full validation batch:

```powershell
python measurement/run_validation.py
```

Or recompute metrics from the saved CSV:

```powershell
python measurement/validate_measurements.py measurement/validation_results.csv
```

Current summary:

```text
Samples:      10
Width MAE:    2.67 mm
Height MAE:   5.22 mm
Overall MAE:  3.95 mm
Overall MPE:  3.20%
```

## Assumptions

- The image contains a visible phone cover and a visible card-sized reference object.
- The phone cover and reference card are approximately coplanar.
- The image is captured with the calibrated camera.
- If `--skip-undistort` is used, the input image is already undistorted.

## Limitations

- Strong perspective tilt can increase measurement error.
- Reference card detection depends on contrast and full card visibility.
- Better camera calibration should improve final measurement accuracy.
