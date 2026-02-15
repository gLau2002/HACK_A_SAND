# Sand & Water Physics with Hand Gestures

Sand and water physics simulation with webcam hand-tracking controls and random disasters (tornado, earthquake, tsunami).

## Requirements

- Python 3
- Webcam
- Dependencies in `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

## Run

```bash
python main_xy.py
```

Requires `assets/BeachBG.png` for the optional beach background. Core physics live in `simulation.py`; `main_xy.py` handles camera, gestures, rendering, and disaster scheduling.

---

## 1. Controls and keybinds

### Keyboard

| Action                        | Key   |
| ----------------------------- | ----- |
| Quit                          | **Q** |
| Toggle background             | **Space** (beach image vs camera display) |
| Toggle hand wireframe         | **H** |
| Toggle drop material          | **Tab** (sand / water; placement is via hand) |
| Queue Tornado (5s warning)    | **1** |
| Queue Earthquake (5s warning) | **2** |
| Queue Tsunami (5s warning)    | **3** |

### Hand gestures (MediaPipe, one hand)

- **Fist** (all fingers curled): **Erase** — clears a rectangular region under the hand (bounds from min/max landmark positions).
- **Thumb + Index pinch** (“grab”): **Grab/drag** — samples a circular region (radius 5 cells) at index tip, clears it, and carries it; releasing places the blob at the new index tip.
- **Thumb + Middle pinch**: Place **sand** at index tip (radius 2 cells, 3 placements per frame).
- **Thumb + Ring pinch**: Place **water** at index tip (radius 2 cells, 3 placements per frame).
- **Thumb + Pinky pinch**: Unused/Reserved for future use.

Pinch is detected when the chosen finger-tip is within 40 px of the thumb tip and other fingertips are farther than 50 px. Thumb must be away from the wrist (not folded) for grab/sand/water to activate.

There is no mouse/click control; input is keyboard + webcam hand tracking.

---

## 2. Cell types and behaviors

Defined in `simulation.py`: `EMPTY`, `SAND`, `WATER`, `WET_SAND`.

| Cell      | ID | Behavior |
| --------- | -- | -------- |
| **EMPTY** | 0  | Passive; other particles can move into it. |
| **SAND**  | 1  | Falls down; if blocked, tries down-left or down-right (random order). Can swap with WATER. When adjacent to WATER (or wet sand connected to dry sand), can be converted to WET_SAND in the absorption pass (batch size 10, depth-limited BFS). |
| **WATER** | 2  | Falls down, then diagonals; then spreads horizontally into EMPTY. Removed when absorbed: if adjacent to SAND or wet-sand-connected-to-dry-sand, one water cell is removed and a patch of SAND becomes WET_SAND (up to 100 water cells per frame). |
| **WET_SAND** | 3 | Same fall rules as SAND but low diagonal-fall probability (2%) when blocked below, so it holds shape better. Water can “reach” dry sand through wet sand via BFS to convert more sand to wet. |

Rendering: EMPTY = black; SAND / WATER / WET_SAND use BGR colors (CELL size 8 px) from `main_xy.py`.

---

## 3. Disaster types

Disaster strikes every 30–45 s, or manually with **1** / **2** / **3**. A 5 s warning is shown, then the effect runs for 60 frames.

| Disaster    | Behavior |
| ----------- | -------- |
| **Tornado** | Vortex around bottom-center. Non-empty cells in a circular region are rotated and lifted; horizontal fling 1.5×. Center drifts randomly; radius and intensity vary per frame. |
| **Earthquake** | Each non-empty cell swaps with a random 4-neighbor. Global shuffle. Screen shake in the app. |
| **Tsunami** | From left or right randomly. Existing SAND/WET_SAND is displaced inward. |

---

## 4. Technologies used

- **Python 3**
- **OpenCV** (`opencv-python`): camera capture, window, drawing, resize, warp for screen shake
- **MediaPipe**: hand landmark detection (21 points); gesture recognition (pinch, fist, fingertip position)
- **NumPy**: grid representation and array ops for simulation and rendering
- **Matplotlib**: in `requirements.txt` (used by `sandTest.py` or tooling; not used in the main app loop)
- **Cursor**: Agentic code development and debugging.

**Assets:** `assets/BeachBG.png` — background when not toggled to camera view.
