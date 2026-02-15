import random
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from simulation import (
    create_grid,
    step,
    place_sand,
    place_water,
    erase_region,
    grab_region,
    place_blob,
    apply_tornado,
    apply_earthquake,
    apply_tsunami,
    SAND,
    WATER,
    WET_SAND,
    EMPTY,
)

# Sand overlay: cell size in pixels (pixelated look), BGR color
CELL = 8  # was 4
WET_SAND_COLOR_BGR = (34, 89, 122)  # tan/beige
DRY_SAND_COLOR_BGR = (80, 187, 229)  # lighter tan
WATER_COLOR_BGR = (246, 118, 86)
# Lookup for vectorized draw: index = cell type (EMPTY=0, SAND=1, WATER=2, WET_SAND=3)
COLORS_BGR = np.array([(0, 0, 0), DRY_SAND_COLOR_BGR, WATER_COLOR_BGR, WET_SAND_COLOR_BGR], dtype=np.uint8)
PINCH_THRESHOLD = 40
NOT_PINCH_THRESHOLD = 50

# MediaPipe Hands landmark indices (21 points)
WRIST = 0
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
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
SAND_SUBSTEPS = 3
PLACE_RADIUS = 2
GRAB_RADIUS = 5
DISASTER_DURATION_FRAMES = 60
WAVE_INTERVAL = 10
DISASTER_MIN_TIME_BETWEEN = 30
DISASTER_MAX_TIME_BETWEEN = 45

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


def start_disaster(disaster: dict, disaster_type: str, cols: int, rows: int) -> None:
    """Start a disaster now: set type, frames_left, and type-specific params (center_col, tsunami_side)."""
    disaster["type"] = disaster_type
    disaster["frames_left"] = DISASTER_DURATION_FRAMES
    if disaster_type == "tornado":
        disaster["center_col"] = random.randint(cols // 5, cols * 4 // 5) if cols > 0 else cols // 2
    elif disaster_type == "tsunami":
        disaster["tsunami_side"] = random.choice(["left", "right"])


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
        num_hands=1,
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
    show_hand_wireframe = True
    drop_mode = "sand"
    disaster = {"type": None, "frames_left": 0, "center_col": 0}
    disaster_queue = []  # list of {"scheduled_at": float, "type": "tornado"|"earthquake"|"tsunami"}
    next_auto_disaster_time = time.time() + random.uniform(DISASTER_MIN_TIME_BETWEEN, DISASTER_MAX_TIME_BETWEEN)
    grabbed_blob = None  # {"cells": [(dr, dc, cell_type), ...]} when dragging
    last_drag_center = (0, 0)  # grid coords for place on release
    prev_gesture = None  # to detect grab transition for re-pinch replace

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
        current_mode = None  # "drag", "grab", "sand", "water", "erase"
        current_tip_px = None
        current_tip_grid = None  # (row, col) for overlay drawing

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                # Hand label ("Left"/"Right") if available
                label = "Hand"
                if result.handedness and len(result.handedness) > i and len(result.handedness[i]) > 0:
                    label = result.handedness[i][0].category_name  # "Left" or "Right"

                # Convert normalized -> pixel coords (OpenCV top-left origin)
                pts = [(lm.x * w, lm.y * h) for lm in hand_lms]

                if show_hand_wireframe:
                    # Draw skeleton
                    for a, b in HAND_CONNECTIONS:
                        draw_line(frame, pts[a], pts[b], color=(0, 255, 0), t=2)
                    for p in pts:
                        draw_point(frame, p, color=(0, 255, 0), r=3)

                # Index fingertip coordinates with bottom-left origin
                tip = hand_lms[INDEX_TIP]  # normalized coords
                x_bl, y_bl = to_bottom_left_coords(tip.x, tip.y, w, h)
                idx_coords[label] = (x_bl, y_bl)
                x_cv = int(tip.x * w)
                y_cv = int(tip.y * h)

                if show_hand_wireframe:
                    # Draw a bigger dot on the index fingertip
                    cv2.circle(frame, (x_cv, y_cv), 8, (0, 255, 0), -1)
                
                
                # fist detection:
                curl_margin = 10  # pixels; tune if needed

                index_curled  = pts[INDEX_TIP][1]  > pts[INDEX_PIP][1]  + curl_margin
                middle_curled = pts[MIDDLE_TIP][1] > pts[MIDDLE_PIP][1] + curl_margin
                ring_curled   = pts[RING_TIP][1]   > pts[RING_PIP][1]   + curl_margin
                pinky_curled  = pts[PINKY_TIP][1]  > pts[PINKY_PIP][1]  + curl_margin

                is_fist = index_curled and middle_curled and ring_curled and pinky_curled

                if is_fist:
                    x_min = min(p[0] for p in pts)
                    x_max = max(p[0] for p in pts)
                    y_min = min(p[1] for p in pts)
                    y_max = max(p[1] for p in pts)
                    rows_g, cols_g = sand_grid.shape
                    row_min = max(0, int(y_min // CELL))
                    row_max = min(rows_g - 1, int(y_max // CELL))
                    col_min = max(0, int(x_min // CELL))
                    col_max = min(cols_g - 1, int(x_max // CELL))
                    erase_region(sand_grid, row_min, row_max, col_min, col_max)

                pinch_px = dist(np.array(pts[THUMB_TIP]), np.array(pts[INDEX_TIP]))

                # pinch distance (thumb tip to index tip) in pixels
                d_index  = dist(np.array(pts[THUMB_TIP]), np.array(pts[INDEX_TIP]))
                d_middle = dist(np.array(pts[THUMB_TIP]), np.array(pts[MIDDLE_TIP]))
                d_ring   = dist(np.array(pts[THUMB_TIP]), np.array(pts[RING_TIP]))
                d_pinky  = dist(np.array(pts[THUMB_TIP]), np.array(pts[PINKY_TIP]))

                # Gesture remap: index=grab/drag, middle=sand, ring=water, pinky=no-op
                dists = {
                    "grab": d_index,   # thumb+index
                    "sand": d_middle,  # thumb+middle
                    "water": d_ring,   # thumb+ring
                    "pinky": d_pinky,  # thumb+pinky (reserved/no-op)
                }

                # Find which finger is closest to the thumb
                closest_name = min(dists, key=dists.get)
                closest_dist = dists[closest_name]

                # check for closest
                others_far = all(
                    d > NOT_PINCH_THRESHOLD
                    for name, d in dists.items()
                    if name != closest_name
                )

                gesture = None
                if closest_dist < PINCH_THRESHOLD and others_far:
                    gesture = closest_name  # "grab" / "sand" / "water" / "pinky"

                # Optional: draw/debug distances on screen
                # cv2.putText(frame, f"dI:{d_index:.0f} dM:{d_middle:.0f} dR:{d_ring:.0f} dP:{d_pinky:.0f}",
                #            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                if (dist(np.array(pts[THUMB_TIP]), np.array(pts[WRIST])) > PINCH_THRESHOLD):
                    if gesture == "grab":
                        grow, gcol = y_cv // CELL, x_cv // CELL
                        if prev_gesture != "grab":
                            blob = grab_region(sand_grid, grow, gcol, GRAB_RADIUS)
                            if blob is not None:
                                grabbed_blob = blob
                        last_drag_center = (grow, gcol)
                    elif gesture == "sand":
                        grow, gcol = y_cv // CELL, x_cv // CELL
                        for _ in range(3):
                            place_sand(sand_grid, grow, gcol, radius=PLACE_RADIUS)
                    elif gesture == "water":
                        grow, gcol = y_cv // CELL, x_cv // CELL
                        for _ in range(3):
                            place_water(sand_grid, grow, gcol, radius=PLACE_RADIUS)
                    elif gesture == "pinky":
                        pass  # reserved for later use

                if gesture != "grab":
                    if grabbed_blob is not None:
                        place_blob(sand_grid, grabbed_blob, last_drag_center[0], last_drag_center[1])
                        grabbed_blob = None

                # Open hand (no pinch, index extended): reserved for future use
                if is_fist or gesture is not None:
                    pass
                elif not index_curled:
                    pass  # open hand, index extended — reserved
                else:
                    pass

                if show_hand_wireframe:
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
                prev_gesture = gesture
                if is_fist:
                    current_mode = "erase"
                elif gesture == "grab":
                    current_mode = "drag" if grabbed_blob else "grab"
                elif gesture == "sand":
                    current_mode = "sand"
                elif gesture == "water":
                    current_mode = "water"
                # open hand (not index_curled, no pinch) — no mode set, reserved for future
                current_tip_px = (x_cv, y_cv)
                current_tip_grid = (y_cv // CELL, x_cv // CELL)
        else:
            prev_gesture = None
            if grabbed_blob is not None:
                place_blob(sand_grid, grabbed_blob, last_drag_center[0], last_drag_center[1])
                grabbed_blob = None

        # Auto-disaster: every 10-60s queue one random disaster (5s from now)
        if sand_grid is not None and time.time() >= next_auto_disaster_time:
            disaster_queue.append({
                "scheduled_at": time.time() + 5,
                "type": random.choice(["tornado", "earthquake", "tsunami"]),
            })
            next_auto_disaster_time = time.time() + random.uniform(DISASTER_MIN_TIME_BETWEEN, DISASTER_MAX_TIME_BETWEEN)

        # Process queue: start any disasters whose scheduled time has passed
        while disaster_queue and time.time() >= disaster_queue[0]["scheduled_at"]:
            entry = disaster_queue.pop(0)
            if sand_grid is not None:
                rows_g, cols_g = sand_grid.shape
                start_disaster(disaster, entry["type"], cols_g, rows_g)

        # Apply disaster effect (if active) then advance sand physics
        if disaster["frames_left"] > 0:
            if disaster["type"] == "tornado":
                rows_g, cols_g = sand_grid.shape
                radius_base, intensity_base = 5, 5
                radius = max(2, radius_base + random.randint(-2, 2))
                disaster["radius"] = radius

                intensity = intensity_base * (0.7 + 0.6 * random.random())
                disaster["center_col"] += random.randint(-4, 4)
                disaster["center_col"] = max(
                    radius, min(cols_g - 1 - radius, disaster["center_col"])
                )
                apply_tornado(sand_grid, disaster["center_col"], radius=radius, intensity=intensity)
            elif disaster["type"] == "earthquake":
                intensity_base = 0.25
                intensity = intensity_base * (0.7 + 0.6 * random.random())
                apply_earthquake(sand_grid, intensity=intensity)
            elif disaster["type"] == "tsunami":
                frames_elapsed = DISASTER_DURATION_FRAMES - disaster["frames_left"]
                if frames_elapsed % WAVE_INTERVAL == 0:
                    height_base = 8
                    height = height_base * (0.7 + 0.6 * random.random())
                    apply_tsunami(
                        sand_grid,
                        side=disaster["tsunami_side"],
                        wave_height=height,
                        wave_width=3,
                    )
            disaster["frames_left"] -= 1

        for i in range(SAND_SUBSTEPS):
            sand_grid = step(sand_grid, run_absorption=(i == SAND_SUBSTEPS - 1))
        rows, cols = sand_grid.shape
        color_grid = COLORS_BGR[sand_grid]
        scaled = cv2.resize(color_grid, (cols * CELL, rows * CELL), interpolation=cv2.INTER_NEAREST)
        rh, rw = min(rows * CELL, h), min(cols * CELL, w)
        region = frame[:rh, :rw]
        non_empty = (scaled[:rh, :rw, 0] != 0) | (scaled[:rh, :rw, 1] != 0) | (scaled[:rh, :rw, 2] != 0)
        region[non_empty] = scaled[:rh, :rw][non_empty]

        # Visual feedback: drag preview (grabbed blob follows pointer), mode indicator
        if current_mode == "drag" and grabbed_blob is not None and current_tip_grid is not None:
            gr0, gc0 = current_tip_grid
            for dr, dc, cell_type in grabbed_blob["cells"]:
                r, c = gr0 + dr, gc0 + dc
                x1, y1 = c * CELL, r * CELL
                x2, y2 = (c + 1) * CELL, (r + 1) * CELL
                if 0 <= x1 < w and 0 <= y1 < h:
                    color = tuple(int(x) for x in COLORS_BGR[cell_type])
                    cv2.rectangle(frame, (x1, y1), (min(x2, w), min(y2, h)), color, -1)
        if current_mode is not None:
            cv2.putText(frame, current_mode.upper(), (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # White wind particles within tornado column (vertical strip)
        if disaster["type"] == "tornado" and disaster["frames_left"] > 0 and disaster.get("radius") is not None:
            center_col = disaster["center_col"]
            r = disaster["radius"]
            width = int(r * 2)  # wider to account for horizontal displacement
            c_min = max(0, center_col - width)
            c_max = min(cols - 1, center_col + width)
            n_pts = 110
            if c_min <= c_max:
                for _ in range(n_pts):
                    gr = random.randint(0, rows - 1)
                    gc = random.randint(c_min, c_max)
                    px, py = gc * CELL + CELL // 2, gr * CELL + CELL // 2
                    cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)

        # Print to terminal every frame (useful for your game code / debugging)
        # Example: Left=(123,456) Right=(800,300)
        #print(f"Left={idx_coords['Left']}  Right={idx_coords['Right']}")

        # Warning: show "Warning: {Disaster} Incoming" at top-center in yellow when next disaster is within 5s
        now = time.time()
        if disaster_queue and disaster_queue[0]["scheduled_at"] - 5 <= now < disaster_queue[0]["scheduled_at"]:
            name = disaster_queue[0]["type"].capitalize()
            warning_text = f"Warning: {name} Incoming"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            (tw, th), _ = cv2.getTextSize(warning_text, font, font_scale, thickness)
            x = (w - tw) // 2
            y = 28
            cv2.putText(frame, warning_text, (x, y), font, font_scale, (0, 255, 255), thickness)

        help_color = (0, 0, 0) if white_background else (255, 255, 255)
        cv2.putText(frame, "Index fingertip (x,y). Bottom-left is (0,0). Press Q to quit",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, help_color, 2)
        # cv2.putText(frame, "Space: toggle beach background",
        #             (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, help_color, 2)
        # cv2.putText(frame, "H: toggle hand wireframe",
        #             (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, help_color, 2)
        cv2.putText(frame, "1/2/3: Queue Tornado / Earthquake / Tsunami (5s warning)",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, help_color, 2)

        # Screen shake for tornado and earthquake
        shake_x, shake_y = 0, 0
        if disaster["frames_left"] > 0 and disaster["type"] in ("tornado", "earthquake"):
            shake_x = random.randint(-3, 3)
            shake_y = random.randint(-3, 3)
        if shake_x != 0 or shake_y != 0:
            M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        cv2.imshow("MediaPipe Hands - Index (bottom-left coords)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            white_background = not white_background
        if key in (ord("h"), ord("H")):
            show_hand_wireframe = not show_hand_wireframe
        if key == 9:  # Tab
            drop_mode = "water" if drop_mode == "sand" else "sand"
        if key == ord("1"):
            disaster_queue.append({"scheduled_at": time.time() + 5, "type": "tornado"})
        if key == ord("2"):
            disaster_queue.append({"scheduled_at": time.time() + 5, "type": "earthquake"})
        if key == ord("3"):
            disaster_queue.append({"scheduled_at": time.time() + 5, "type": "tsunami"})
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
