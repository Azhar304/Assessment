import os
import cv2
from cv2 import aruco

# Number of chessboard squares
SQUARES_X = 7
SQUARES_Y = 10

# Physical dimensions (used later for calibration)
SQUARE_LENGTH = 25      # mm
MARKER_LENGTH = 18      # mm

# Print resolution
DPI = 300

# Create ChArUco Board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

board = aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH / 1000.0,     # meters
    MARKER_LENGTH / 1000.0,
    dictionary
)

# Convert mm -> pixels
px_per_mm = DPI / 25.4

board_width_px = int(SQUARES_X * SQUARE_LENGTH * px_per_mm)
board_height_px = int(SQUARES_Y * SQUARE_LENGTH * px_per_mm)

# Generate the image matrix
img = board.generateImage(
    (board_width_px, board_height_px),
    marginSize=20,
    borderBits=1
)

# Ensure the 'calibration' directory exists
output_dir = "calibration"
os.makedirs(output_dir, exist_ok=True)

# Define the full output path
output_path = os.path.join(output_dir, "charuco_board.png")

# Save the image inside the folder
cv2.imwrite(output_path, img)

print("Generated:")
print(f" {output_path}")