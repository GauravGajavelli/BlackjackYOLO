"""
app.py — Flask backend for the web-based Blackjack Bot.

Serves the game UI and exposes an API for YOLO card detection,
basic strategy lookup, and Hi-Lo card counting.
"""

import argparse
import base64
import io

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from bot.detector import CardDetector
from bot.strategy import HiLoCounter, basic_strategy, hand_total

parser = argparse.ArgumentParser(description="Blackjack Bot server")
parser.add_argument("--pipeline", action="store_true",
                    help="Use YOLO+ResNet18 two-stage pipeline for better accuracy")
parser.add_argument("--debug", action="store_true",
                    help="Save detection images to debug/ folder for inspection")
args = parser.parse_args()

if args.debug:
    import os, time as _time
    DEBUG_DIR = "debug_frames"
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"[app] Debug mode enabled — saving detection images to {DEBUG_DIR}/", flush=True)

app = Flask(__name__, static_folder="game")

# ── Load model and state at startup ──────────────────────────────────────────
detector = CardDetector(pipeline=args.pipeline)
counter = HiLoCounter()
prev_seen: set[str] = set()  # cards seen this hand (avoid double-counting)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the blackjack game page."""
    return send_from_directory("game", "blackjack.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Receive a base64 PNG screenshot, run YOLO detection, compute strategy.

    Request JSON: { "image": "<base64-encoded PNG>" }
    Response JSON: {
        "detections": [{ "class_name", "confidence", "bbox", "zone" }, ...],
        "dealer_cards": [...], "player_cards": [...],
        "dealer_total": int, "player_total": int,
        "action": str,
        "running_count": int, "true_count": float, "cards_seen": int
    }
    """
    data = request.get_json(force=True)
    image_b64 = data.get("image", "")

    # Strip data-URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    # Decode base64 → PIL → BGR numpy array
    raw = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Run detection (YOLO or pipeline) with iterative masking
    detections = detector.detect_with_masking(frame)
    hands = detector.parse_hands(detections)

    # Debug: save frame and log detections
    if args.debug:
        ts = int(_time.time() * 1000)
        debug_path = f"{DEBUG_DIR}/frame_{ts}.png"
        saved = cv2.imwrite(debug_path, frame)
        print(f"[debug] {'Saved' if saved else 'FAILED to save'} {debug_path} | shape={frame.shape} | detections={len(detections)}", flush=True)
        for det in detections:
            print(f"[debug]   {det['class_name']} conf={det['confidence']:.3f} bbox={det['bbox']}", flush=True)
        if not detections:
            print("[debug]   No detections — check saved image for card visibility", flush=True)

    # Update Hi-Lo count with newly seen cards
    global prev_seen
    all_cards = hands["dealer"] + hands["player"]
    new_cards = [c for c in all_cards if c not in prev_seen]
    if new_cards:
        counter.update(new_cards)
        prev_seen.update(new_cards)

    # Compute strategy
    action = "none"
    dealer_total_val = 0
    player_total_val = 0

    if hands["player"]:
        player_total_val, _ = hand_total(hands["player"])

    if hands["dealer"]:
        dealer_total_val, _ = hand_total(hands["dealer"])

    if hands["player"] and hands["dealer"]:
        can_double = len(hands["player"]) == 2
        action = basic_strategy(
            hands["player"],
            hands["dealer"][0],  # upcard
            can_double=can_double,
            can_split=can_double,
        )

    # Build response detections with zone info and pass number
    det_response = []
    total_passes = 0
    for det in detections:
        cy = det["center"][1]
        if det["class_name"] in hands["dealer"]:
            zone = "dealer"
        elif det["class_name"] in hands["player"]:
            zone = "player"
        else:
            zone = "unknown"
        pass_num = det.get("pass_num", 1)
        total_passes = max(total_passes, pass_num)
        det_response.append({
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 2),
            "bbox": det["bbox"],
            "zone": zone,
            "pass_num": pass_num,
        })

    return jsonify({
        "detections": det_response,
        "dealer_cards": hands["dealer"],
        "player_cards": hands["player"],
        "dealer_total": dealer_total_val,
        "player_total": player_total_val,
        "action": action,
        "running_count": counter.running_count,
        "true_count": round(counter.true_count, 1),
        "cards_seen": counter.cards_seen,
        "total_passes": total_passes,
    })


@app.route("/api/strategy", methods=["POST"])
def strategy():
    """
    Receive card names directly from JS game state, return strategy action.

    Request JSON: { "player_cards": ["Ks","7h"], "dealer_upcard": "10c", "can_double": true }
    """
    data = request.get_json(force=True)
    player_cards = data["player_cards"]
    dealer_upcard = data["dealer_upcard"]
    can_double = data.get("can_double", True)

    # Update count with newly seen cards
    global prev_seen
    all_cards = player_cards + [dealer_upcard]
    new_cards = [c for c in all_cards if c not in prev_seen]
    if new_cards:
        counter.update(new_cards)
        prev_seen.update(new_cards)

    p_total, _ = hand_total(player_cards)
    d_total, _ = hand_total([dealer_upcard])

    action = basic_strategy(
        player_cards, dealer_upcard,
        can_double=can_double, can_split=can_double,
    )

    return jsonify({
        "action": action,
        "player_total": p_total,
        "dealer_total": d_total,
        "running_count": counter.running_count,
        "true_count": round(counter.true_count, 1),
        "cards_seen": counter.cards_seen,
    })


@app.route("/api/new_hand", methods=["POST"])
def new_hand():
    """Clear prev_seen for a new hand (keep the running count)."""
    global prev_seen
    prev_seen = set()
    return jsonify({"status": "ok"})


@app.route("/api/reset_count", methods=["POST"])
def reset_count():
    """Reset the Hi-Lo counter for a new shoe."""
    global prev_seen
    counter.reset()
    prev_seen = set()
    return jsonify({"status": "ok", "running_count": 0, "true_count": 0.0})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
