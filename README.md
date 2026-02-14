# Blackjack Card Counting Bot — YOLO + Screen Automation

A real-time blackjack bot that uses a YOLO object detection model to read cards
from a browser-based blackjack game, then makes optimal decisions using Basic
Strategy and Hi-Lo card counting.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Browser Window                      │
│  ┌───────────────────────────────────────────────┐  │
│  │         game/blackjack.html                   │  │
│  │   (self-hosted HTML5 blackjack game)          │  │
│  └───────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │ screen capture (mss)
                       ▼
┌──────────────────────────────────────────────────────┐
│                  bot/main.py                          │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ capture.py │→ │ detector.py │→ │ strategy.py  │  │
│  │ (grab      │  │ (YOLO       │  │ (basic strat │  │
│  │  screen)   │  │  inference)  │  │  + Hi-Lo)    │  │
│  └────────────┘  └─────────────┘  └──────┬───────┘  │
│                                          │           │
│  ┌────────────┐  ┌─────────────┐         │           │
│  │ overlay.py │  │ clicker.py  │◄────────┘           │
│  │ (show YOLO │  │ (pyautogui  │                     │
│  │  boxes)    │  │  clicks)    │                     │
│  └────────────┘  └─────────────┘                     │
└──────────────────────────────────────────────────────┘
```

### Data Flow (each game step)

1. **Capture** — `mss` grabs a screenshot of the browser region
2. **Detect** — YOLO model predicts bounding boxes + card classes
3. **Parse** — Detections are mapped to dealer vs. player hands using spatial regions
4. **Decide** — Basic Strategy table + Hi-Lo running count → action
5. **Act** — `pyautogui` clicks the correct button (Hit / Stand / Double / etc.)
6. **Overlay** — OpenCV window shows the annotated frame (for demo/audience)

## Project Structure

```
blackjack-bot/
├── README.md                 # You are here
├── requirements.txt          # pip dependencies
├── game/
│   └── blackjack.html        # Self-hosted browser blackjack game
├── bot/
│   ├── main.py               # Entry point — game loop
│   ├── capture.py            # Screen capture utilities
│   ├── detector.py           # YOLO inference wrapper
│   ├── strategy.py           # Basic Strategy + Hi-Lo counting
│   └── clicker.py            # Mouse automation (clicking buttons)
├── config/
│   └── settings.py           # All configurable coordinates & parameters
└── utils/
    ├── calibrate.py          # Interactive tool to find screen coords
    └── overlay.py            # Draw YOLO detections for demo display
```

## Setup

### 1. Python environment

Requires **Python 3.9+**. A virtual environment is recommended:

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. YOLO model weights

Place your trained YOLO weights file (e.g. `best.pt`) in the project root or
update the path in `config/settings.py`:

```python
MODEL_PATH = "best.pt"   # ← point this to your weights
```

### 3. Launch the game

Open `game/blackjack.html` in your browser. **Do not resize or move the
window** during a session. Position it in a consistent spot on your screen.

### 4. Calibrate screen regions

Run the calibration tool once to measure coordinates:

```bash
python -m utils.calibrate
```

This will let you:
- Click corners of the game area → sets `MONITOR_REGION`
- Click the center of each button (Hit, Stand, Double, Deal) → sets button coords
- Define dealer-zone and player-zone y-boundaries

Copy the printed values into `config/settings.py`.

### 5. Run the bot

```bash
python -m bot.main
```

Controls:
- **Space** — trigger one decision step (recommended for demos)
- **R** — run continuously (auto-play)
- **Q** — quit
- **C** — print current Hi-Lo count

## Dependencies

| Package       | Purpose                        |
|---------------|--------------------------------|
| ultralytics   | YOLO model loading & inference |
| mss           | Fast screen capture            |
| pyautogui     | Mouse clicks & movement        |
| opencv-python | Image display & annotation     |
| numpy         | Array operations               |
| keyboard      | Global hotkey listener         |

## YOLO Class Mapping

The detector expects your YOLO model to output class names like:
`2c`, `2d`, `2h`, `2s`, `3c`, ..., `As`, `Ah`, `Ad`, `Ac`

Where the first character(s) are the rank (2-10, J, Q, K, A) and the last
character is the suit (c=clubs, d=diamonds, h=hearts, s=spades).

Update the `CLASS_NAMES` list in `config/settings.py` if your model uses a
different naming convention.

## Tips for the Class Demo

1. Run in **step mode** (Space to advance) so the audience can see each detection.
2. The overlay window shows live bounding boxes — point your projector at that.
3. Ask classmates to guess the action before you press Space.
4. Show a mix of successes and failures — the failures are more interesting!
5. Keep the browser and overlay windows side-by-side on screen.
