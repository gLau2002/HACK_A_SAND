import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from simulation import create_grid, step, place_sand, place_water, SAND, WATER, WET_SAND, EMPTY

# Sand overlay: cell size in pixels (pixelated look), BGR color
CELL = 4
WET_SAND_COLOR_BGR = (34, 89, 122)  # tan/beige
DRY_SAND_COLOR_BGR = (80, 187, 229)  # lighter tan
WATER_COLOR_BGR = (246, 118, 86)
# Lookup for vectorized draw: index = cell type (EMPTY=0, SAND=1, WATER=2, WET_SAND=3)
COLORS_BGR = np.array([(0, 0, 0), DRY_SAND_COLOR_BGR, WATER_COLOR_BGR, WET_SAND_COLOR_BGR], dtype=np.uint8)
PINCH_THRESHOLD = 35
SAND_SUBSTEPS = 3
PLACE_RADIUS = 2

# -----------------------------
# Helpers
# -----------------------------
def dist(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def draw_point(img, p, color=(0, 255, 0), r=4):
    cv2.circle(img, (int(p[0]), int(p[1])), r, color, -1)


def draw_line(img, p1, p2, color=(0, 255, 0), t=2):
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, t)


def to_bottom_left_coords(norm_x: float, norm_y: float, w: int, h: int):
    """
    MediaPipe gives normalized coords where (0,0) is top-left.
    Convert to pixel coords where (0,0) is bottom-left.
    """
    x_px = int(norm_x * w)
    y_top = int(norm_y * h)
    y_bottom = (h - 1) - y_top
    return x_px, y_bottom


# MediaPipe Hands landmark indices (21 points)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# A simple skeleton (connections) for the hand
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky

    # Knuckle connections across the palm
    (5, 9), (9, 13), (13, 17)
]


def main():
    # -----------------------------
    # Webcam setup
    # -----------------------------
    cam_index = 0  # try 0/1/2 if needed
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}. Try 0, 1, 2...")

    # -----------------------------
    # MediaPipe Tasks: HandLandmarker
    # -----------------------------
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = str(Path(__file__).with_name("hand_landmarker.task"))

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    beach_bg_path = Path(__file__).parent / "assets" / "BeachBG.png"
    beach_bg_img = cv2.imread(str(beach_bg_path))
    if beach_bg_img is None:
        raise FileNotFoundError(f"Could not load background image: {beach_bg_path}")

    sand_grid = None  # created on first frame when we have h, w
    white_background = False
    drop_mode = "sand"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror so it feels like "selfie view"
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if sand_grid is None:
            sand_grid = create_grid(h // CELL, w // CELL)

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, t_ms)

        if white_background:
            frame = cv2.resize(beach_bg_img, (w, h), interpolation=cv2.INTER_NEAREST).copy()

        # We'll store index fingertip coords for each hand (bottom-left origin)
        idx_coords = {"Left": None, "Right": None}

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                # Hand label ("Left"/"Right") if available
                label = "Hand"
                if result.handedness and len(result.handedness) > i and len(result.handedness[i]) > 0:
                    label = result.handedness[i][0].category_name  # "Left" or "Right"

                # Convert normalized -> pixel coords (OpenCV top-left origin)
                pts = [(lm.x * w, lm.y * h) for lm in hand_lms]

                # Draw skeleton
                for a, b in HAND_CONNECTIONS:
                    draw_line(frame, pts[a], pts[b], color=(0, 255, 0), t=2)
                for p in pts:
                    draw_point(frame, p, color=(0, 255, 0), r=3)

                # Index fingertip coordinates with bottom-left origin
                tip = hand_lms[INDEX_TIP]  # normalized coords
                x_bl, y_bl = to_bottom_left_coords(tip.x, tip.y, w, h)
                idx_coords[label] = (x_bl, y_bl)

                # Draw a bigger dot on the index fingertip
                x_cv = int(tip.x * w)
                y_cv = int(tip.y * h)
                cv2.circle(frame, (x_cv, y_cv), 8, (0, 255, 0), -1)

                # Optional: pinch distance (thumb tip to index tip) in pixels
                pinch_px = dist(np.array(pts[THUMB_TIP]), np.array(pts[INDEX_TIP]))

                # Spawn pixelated sand or water at fingertip when pinch < 35
                if pinch_px < PINCH_THRESHOLD:
                    grow, gcol = y_cv // CELL, x_cv // CELL
                    for _ in range(3):
                        if drop_mode == "sand":
                            place_sand(sand_grid, grow, gcol, radius=PLACE_RADIUS)
                        else:
                            place_water(sand_grid, grow, gcol, radius=PLACE_RADIUS)

                # Text near fingertip (shows bottom-left coords)
                cv2.putText(
                    frame,
                    f"{label} idx: ({x_bl}, {y_bl})  pinch:{pinch_px:0.1f}px",
                    (x_cv + 10, y_cv - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Advance sand physics and draw pixelated sand on frame (absorption only on last substep)
        for i in range(SAND_SUBSTEPS):
            sand_grid = step(sand_grid, run_absorption=(i == SAND_SUBSTEPS - 1))
        rows, cols = sand_grid.shape
        color_grid = COLORS_BGR[sand_grid]
        scaled = cv2.resize(color_grid, (cols * CELL, rows * CELL), interpolation=cv2.INTER_NEAREST)
        rh, rw = min(rows * CELL, h), min(cols * CELL, w)
        region = frame[:rh, :rw]
        non_empty = (scaled[:rh, :rw, 0] != 0) | (scaled[:rh, :rw, 1] != 0) | (scaled[:rh, :rw, 2] != 0)
        region[non_empty] = scaled[:rh, :rw][non_empty]

        # Print to terminal every frame (useful for your game code / debugging)
        # Example: Left=(123,456) Right=(800,300)
        print(f"Left={idx_coords['Left']}  Right={idx_coords['Right']}")

        help_color = (0, 0, 0) if white_background else (255, 255, 255)
        cv2.putText(frame, "Index fingertip (x,y). Bottom-left is (0,0). Press Q to quit",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, help_color, 2)
        cv2.putText(frame, "Space: toggle beach background",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, help_color, 2)
        cv2.putText(frame, "Tab: toggle sand/water",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, help_color, 2)

        cv2.imshow("MediaPipe Hands - Index (bottom-left coords)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            white_background = not white_background
        if key == 9:  # Tab
            drop_mode = "water" if drop_mode == "sand" else "sand"
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
