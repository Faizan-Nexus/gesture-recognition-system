"""
Configuration constants for the gesture recognition system.
"""

# MediaPipe Hands Configuration
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Gesture Matching Configuration
MATCH_THRESHOLD = 0.23  # Distance threshold for gesture acceptance
SMOOTHING_BUFFER_SIZE = 5  # Number of frames for majority voting

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Calibration Configuration
CALIBRATION_FRAMES = 25  # Number of frames to collect during calibration
CALIBRATION_GESTURE = "open_palm"  # Reference gesture for calibration

# Gesture Names
GESTURE_NAMES = [
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "thumbs_down",
    "pointing",
    "pinch",
    "ok_sign",
    "call_me",
    "palm_up",
    "peace_sign"
]

# Display Configuration
DISPLAY_WINDOW_NAME = "Gesture Recognition System"
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  # Green
LANDMARK_COLOR = (255, 0, 0)  # Blue (BGR format)
CONNECTION_COLOR = (0, 255, 0)  # Green
