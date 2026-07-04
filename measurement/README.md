# Measurement Pipeline

This folder contains Step 3 code for converting phone-cover segmentation masks into real-world measurements.

## Method

The first implementation uses a known-size reference object in the same image. The user provides the reference bounding box and the real reference dimensions in millimetres.

Pipeline:

```text
raw image
  -> camera undistortion
  -> Mask R-CNN segmentation
  -> contour extraction from predicted mask
  -> reference pixels-per-mm conversion
  -> width/height measurement in mm
```

## Measure One Image

```powershell
python measurement/measure_object.py path/to/image.jpg --reference-bbox x,y,width,height --reference-width-mm 85.6 --reference-height-mm 54.0
```

Use `--skip-undistort` only when the input image is already undistorted.

Outputs:

```text
measurement/outputs/<image>_measurement.jpg
measurement/outputs/<image>_measurement.json
```

## Validate Accuracy

Fill `measurement/validation_template.csv` with 10+ physical measurements and model predictions, then run:

```powershell
python measurement/validate_measurements.py measurement/validation_template.csv
```

Output:

```text
measurement/validation_summary.json
```

The summary reports mean absolute error (MAE) and mean percentage error (MPE), as required by the assessment.
