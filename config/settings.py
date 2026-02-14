"""
config/settings.py — All tunable parameters in one place.
"""

# ---------------------------------------------------------------------------
# YOLO Model
# ---------------------------------------------------------------------------
MODEL_PATH = "models/best.pt"    # Path to your trained YOLO weights
CONFIDENCE_THRESHOLD = 0.25      # Minimum detection confidence to accept
IOU_THRESHOLD = 0.40             # NMS IoU threshold

# ---------------------------------------------------------------------------
# Spatial Zones — dividing dealer vs. player cards
# ---------------------------------------------------------------------------
# These are relative to the #table div captured by html2canvas (820x520).
DEALER_ZONE_Y = (20, 200)       # (top, bottom) of dealer card region
PLAYER_ZONE_Y = (320, 500)      # (top, bottom) of player card region

# ---------------------------------------------------------------------------
# Card Class Names
# ---------------------------------------------------------------------------
# Must match the class names your YOLO model was trained with.
# Format: "<rank><suit>" where suit is one of c, d, h, s
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]   # clubs, diamonds, hearts, spades

CLASS_NAMES = [f"{r}{s}" for r in RANKS for s in SUITS]  # 52 classes

# ---------------------------------------------------------------------------
# Hi-Lo Counting
# ---------------------------------------------------------------------------
DECKS_IN_SHOE = 6               # Typical shoe size; adjust for your game

# ---------------------------------------------------------------------------
# Iterative Masking Detection
# ---------------------------------------------------------------------------
MAX_DETECTION_PASSES = 3        # Max YOLO passes with masking
MASK_PADDING = 5                # Pixels to pad around bbox when masking
DEDUP_IOU_THRESHOLD = 0.50      # IoU threshold for duplicate filtering

# ---------------------------------------------------------------------------
# Two-Stage Pipeline (YOLO + ResNet18)
# ---------------------------------------------------------------------------
PIPELINE_MODE = False
CNN_MODEL_PATH = "models/cnn_resnet18_224x224.pt"
CNN_INPUT_SIZE = (224, 224)
CNN_CONFIDENCE_THRESHOLD = 0.30
CNN_CLASS_NAMES = sorted(CLASS_NAMES)  # alphabetical — matches ImageFolder training order
