"""Shared utilities for Mask R-CNN model loading and image undistortion."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = PROJECT_ROOT / "calibration" / "results"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "maskrcnn.pth"
NUM_CLASSES = 2


def load_calibration():
    """Load camera matrix and distortion coefficients."""
    camera_matrix = np.load(CALIB_DIR / "camera_matrix.npy")
    dist_coeffs = np.load(CALIB_DIR / "dist_coeffs.npy")
    return camera_matrix, dist_coeffs


def undistort_image(image, camera_matrix=None, dist_coeffs=None):
    """Undistort a BGR image using stored calibration parameters."""
    if camera_matrix is None or dist_coeffs is None:
        camera_matrix, dist_coeffs = load_calibration()

    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(
        image, camera_matrix, dist_coeffs, None, new_camera_matrix
    )
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y : y + rh, x : x + rw]
    return undistorted


def build_model(num_classes=NUM_CLASSES):
    """Build Mask R-CNN with custom heads for phone cover segmentation."""
    model = maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    return model


def load_model(weights_path=None, device=None):
    """Load trained Mask R-CNN weights."""
    if weights_path is None:
        weights_path = DEFAULT_WEIGHTS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device
