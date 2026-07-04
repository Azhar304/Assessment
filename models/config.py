"""
============================================================
Configuration
Phone Cover Instance Segmentation
============================================================
"""


# Project

OBJECT_CLASS = "Phone_Cover"

FRAMEWORK = "pytorch_torchvision"

TRAINING_ENVIRONMENT = "google_colab_gpu"



# Dataset

DATASET_FORMAT = "coco_instance_segmentation"

DRIVE_ROOT = "/content/drive/MyDrive/dataset"

TRAIN_IMAGES = "/content/drive/MyDrive/dataset/train/images"
TRAIN_ANNOTATIONS = "/content/drive/MyDrive/dataset/train/annotations.json"
TRAIN_COUNT = 62

VAL_IMAGES = "/content/drive/MyDrive/dataset/val/images"
VAL_ANNOTATIONS = "/content/drive/MyDrive/dataset/val/annotations.json"
VAL_COUNT = 19

TEST_IMAGES = "/content/drive/MyDrive/dataset/test/images"
TEST_ANNOTATIONS = "/content/drive/MyDrive/dataset/test/annotations.json"
TEST_COUNT = 9



# Model

MODEL_NAME = "maskrcnn_resnet50_fpn"

PRETRAINED_WEIGHTS = "DEFAULT"

NUM_CLASSES = 2

CLASS_NAMES = [
    "background",
    "Phone_Cover"
]

BOX_PREDICTOR = "FastRCNNPredictor"

MASK_PREDICTOR = "MaskRCNNPredictor"

MASK_HIDDEN_LAYER = 256



# Training

NUM_EPOCHS = 15

TRAIN_BATCH_SIZE = 2

VAL_BATCH_SIZE = 1

TEST_BATCH_SIZE = 1

LEARNING_RATE = 0.005

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0005

OPTIMIZER = "SGD"

LR_SCHEDULER = "StepLR"

STEP_SIZE = 10

GAMMA = 0.1

RANDOM_SEED = 42

# Inference

CONFIDENCE_THRESHOLD = 0.50

MASK_THRESHOLD = 0.50

# Outputs

NOTEBOOK_PATH = "models/mask_rcnn.ipynb"

TRAINING_HISTORY = "models/training_history.json"

MODEL_WEIGHTS = "models/maskrcnn.pth"

GIT_POLICY = "keep_large_weights_in_drive_not_git"


# Training Results


EPOCHS_COMPLETED = 15

FINAL_TRAIN_LOSS = 0.062234348106768825

FINAL_VALIDATION_LOSS = 0.08831389111123587

BEST_VALIDATION_LOSS = 0.0873759237951354

BEST_VALIDATION_EPOCH = 13


# Evaluation Metrics (test set, 9 images)

TEST_MAP_50 = 1.0

TEST_MAP_50_95 = 0.9523

TEST_MEAN_MASK_IOU = 0.9565

TEST_PRECISION = 1.0

TEST_RECALL = 1.0

TEST_F1 = 1.0