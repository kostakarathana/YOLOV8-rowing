"""
YOLOv8 Rowing Posture Detector
Plays a video with real-time pose estimation overlay and rolling angle graphs.

Usage:
    python detect_posture.py                    # defaults to IMG_0145.MOV
    python detect_posture.py --video path.mov   # custom video
    python detect_posture.py --save             # save annotated output to file

Controls:
    SPACE      - pause / resume
    R          - replay from beginning
    Q/ESC      - quit
    LEFT/RIGHT - step back / forward one frame (auto-pauses)
    S          - screenshot current frame
    +/-        - adjust detection confidence threshold
    TAB        - cycle tracked rower (persistent ID via ByteTrack)
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --- COCO Keypoint Indices ---
# 0:nose 1:L_eye 2:R_eye 3:L_ear 4:R_ear
# 5:L_shoulder 6:R_shoulder 7:L_elbow 8:R_elbow
# 9:L_wrist 10:R_wrist 11:L_hip 12:R_hip
# 13:L_knee 14:R_knee 15:L_ankle 16:R_ankle

SKELETON = [
    (5, 6),   (5, 7),  (7, 9),   (6, 8),  (8, 10),
    (5, 11),  (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]

SEGMENT_COLORS = {
    "torso":     (0, 255, 200),
    "left_arm":  (255, 150, 0),
    "right_arm": (0, 150, 255),
    "left_leg":  (255, 0, 150),
    "right_leg": (0, 255, 100),
}

GRAPH_COLOR = (0, 200, 255)       # orange — tracked rower
GRAPH_COLOR_DIM = (0, 67, 85)     # dim orange — boat average

# Graph settings
GRAPH_PANEL_H = 260
GRAPH_PADDING = 10
GRAPH_HISTORY = 150  # frames of history to show


def segment_color(i, j):
    if i in (5, 7, 9) and j in (5, 7, 9):
        return SEGMENT_COLORS["left_arm"]
    if i in (6, 8, 10) and j in (6, 8, 10):
        return SEGMENT_COLORS["right_arm"]
    if i in (11, 13, 15) and j in (11, 13, 15):
        return SEGMENT_COLORS["left_leg"]
    if i in (12, 14, 16) and j in (12, 14, 16):
        return SEGMENT_COLORS["right_leg"]
    return SEGMENT_COLORS["torso"]


def compute_angle(a, b, c):
    """Angle in degrees at vertex b, given 2D points a-b-c."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return round(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def kp_visible(kp, indices, conf, min_conf=0.3):
    """Check all keypoint indices have sufficient confidence."""
    return all(conf[i] > min_conf and kp[i][0] > 0 for i in indices)


def compute_back_angle(kp, conf):
    """Compute back angle (shoulder-hip-knee) averaging left/right sides.

    Returns angle in degrees rounded to nearest integer, or None if not visible.
    ~30-45 deg at the catch (compressed), ~100-120 deg at the finish (opened up).
    """
    vals = []
    if kp_visible(kp, [5, 11, 13], conf):
        vals.append(compute_angle(kp[5][:2], kp[11][:2], kp[13][:2]))
    if kp_visible(kp, [6, 12, 14], conf):
        vals.append(compute_angle(kp[6][:2], kp[12][:2], kp[14][:2]))
    if vals:
        return round(sum(vals) / len(vals))
    return None


def draw_skeleton(frame, kp, conf, highlight=False):
    """Draw skeleton on frame. If highlight, use thicker lines."""
    pts = kp[:, :2].astype(int)
    thickness = 3 if highlight else 2
    dot_r = 5 if highlight else 3

    for i, j in SKELETON:
        if conf[i] > 0.3 and conf[j] > 0.3:
            color = segment_color(i, j)
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness, cv2.LINE_AA)

    for idx in range(len(pts)):
        if conf[idx] > 0.3 and pts[idx][0] > 0:
            cv2.circle(frame, tuple(pts[idx]), dot_r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[idx]), dot_r, (0, 0, 0), 1, cv2.LINE_AA)


def draw_angle_label(frame, kp, conf, indices, value, color):
    """Draw an angle value near the vertex joint."""
    vertex = indices[1]
    if conf[vertex] > 0.3:
        x, y = int(kp[vertex][0]), int(kp[vertex][1])
        label = f"{value} deg"
        cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)


def _plot_line(panel, values, x0, y0, x1, y1, y_min, y_max, graph_w, graph_h, color, thickness):
    """Helper: draw a line series onto a graph region."""
    points = []
    for i, v in enumerate(values):
        if v is not None:
            px = x0 + int(i / max(GRAPH_HISTORY - 1, 1) * graph_w)
            frac = (v - y_min) / (y_max - y_min)
            py = y1 - int(frac * graph_h)
            py = max(y0, min(y1, py))
            points.append((px, py))
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(panel, points[i], points[i + 1], color, thickness, cv2.LINE_AA)


def draw_graph_panel(panel_w, history, boat_avg_history, fps_video):
    """Draw a single full-width back angle graph with boat average overlay."""
    panel = np.zeros((GRAPH_PANEL_H, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    x0 = GRAPH_PADDING
    y0 = GRAPH_PADDING + 20
    x1 = panel_w - GRAPH_PADDING
    y1 = GRAPH_PANEL_H - GRAPH_PADDING
    graph_w = x1 - x0
    graph_h = y1 - y0
    y_min, y_max = 0, 180

    # Background
    cv2.rectangle(panel, (x0, y0), (x1, y1), (50, 50, 50), -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (80, 80, 80), 1)

    # Title + legend
    cv2.putText(panel, "Back Angle (shoulder-hip-knee)", (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAPH_COLOR, 1, cv2.LINE_AA)
    cv2.putText(panel, "-- boat avg", (x0 + 260, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAPH_COLOR_DIM, 1, cv2.LINE_AA)

    # Y-axis labels
    for deg in range(0, 181, 30):
        frac = deg / y_max
        py = y1 - int(frac * graph_h)
        cv2.line(panel, (x0, py), (x1, py), (60, 60, 60), 1)
        label = f"{deg} deg"
        cv2.putText(panel, label, (x0 + 3, py - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (120, 120, 120), 1, cv2.LINE_AA)

    # Plot boat average (behind, dimmer)
    avg_values = list(boat_avg_history)
    _plot_line(panel, avg_values, x0, y0, x1, y1, y_min, y_max,
               graph_w, graph_h, GRAPH_COLOR_DIM, 1)

    # Plot tracked rower (foreground, bright)
    values = list(history)
    _plot_line(panel, values, x0, y0, x1, y1, y_min, y_max,
               graph_w, graph_h, GRAPH_COLOR, 2)

    # Current value readout (top-right of graph)
    if values and values[-1] is not None:
        cv2.putText(panel, f"{values[-1]} deg", (x1 - 120, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if avg_values and avg_values[-1] is not None:
        cv2.putText(panel, f"avg {avg_values[-1]} deg", (x1 - 120, y0 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAPH_COLOR_DIM, 1, cv2.LINE_AA)

    # Time axis label
    if history:
        secs = len(history) / max(fps_video, 1)
        cv2.putText(panel, f"{secs:.1f}s", (x1 - 40, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1, cv2.LINE_AA)

    return panel


def draw_hud(frame, fps, conf_thresh, state, frame_num, total_frames, fps_video,
             track_id, n_visible):
    """Draw heads-up display with status and controls."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    progress = frame_num / max(total_frames, 1)
    t_cur = frame_num / max(fps_video, 1)
    t_tot = total_frames / max(fps_video, 1)
    status = state.upper()
    id_label = f"ID:{track_id}" if track_id is not None else "none"
    info = (f"{status} | {t_cur:.1f}s / {t_tot:.1f}s | "
            f"Frame {frame_num}/{total_frames} | FPS: {fps:.0f} | "
            f"Conf: {conf_thresh:.2f} | Rower {id_label} ({n_visible} visible)")
    cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    bar_y = 32
    bar_w = int(w * progress)
    cv2.line(frame, (0, bar_y), (w, bar_y), (60, 60, 60), 3)
    cv2.line(frame, (0, bar_y), (bar_w, bar_y), (0, 200, 255), 3)

    controls = "SPACE: pause  R: replay  LEFT/RIGHT: frame step  TAB: switch rower  Q: quit"
    cv2.putText(frame, controls, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (150, 150, 150), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Rowing Posture Detector")
    parser.add_argument("--video", type=str, default="IMG_0145.MOV",
                        help="Path to video file")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt",
                        help="YOLOv8 pose model (default: small for better rowing detection)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25, lower catches more rowers)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated video to output file")
    parser.add_argument("--output", type=str, default="output_detected.mp4",
                        help="Output file path (used with --save)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print("Model loaded.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {vid_w}x{vid_h} @ {fps_video:.0f}fps, {total_frames} frames")

    # Display dimensions
    disp_w = min(vid_w, 1280)
    scale = disp_w / vid_w
    disp_h = int(vid_h * scale)
    combined_h = disp_h + GRAPH_PANEL_H

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_video, (disp_w, combined_h))
        print(f"Saving output to: {args.output}")

    # State
    conf_thresh = args.conf
    paused = False
    ended = False
    frame_num = 0
    frame = None
    fps_actual = 0.0
    elapsed = 0.0
    tracked_id = None       # persistent ByteTrack ID for the rower we're following
    visible_ids = []        # currently visible track IDs (sorted by bbox area, largest first)
    frames_lost = 0         # how many frames the tracked_id has been missing
    MAX_LOST_FRAMES = 30    # after this many lost frames, auto-select a new rower
    angle_history = deque(maxlen=GRAPH_HISTORY)
    boat_avg_history = deque(maxlen=GRAPH_HISTORY)  # average of all OTHER rowers
    window_name = "YOLOv8 Rowing Posture Detector"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, combined_h)

    def reset_tracker():
        """Reset ByteTrack state (call on seek/replay)."""
        nonlocal tracked_id, visible_ids, frames_lost
        if hasattr(model, "predictor") and hasattr(model.predictor, "trackers"):
            for t in model.predictor.trackers:
                t.reset()
        tracked_id = None
        visible_ids = []
        frames_lost = 0

    def process_frame(f):
        """Run tracking + detection, draw overlays, update angle history."""
        nonlocal fps_actual, elapsed, tracked_id, visible_ids, frames_lost

        t0 = time.time()
        results = model.track(f, conf=conf_thresh, persist=True, verbose=False)
        elapsed = time.time() - t0
        fps_actual = 1.0 / max(elapsed, 1e-6)

        result = results[0]
        has_detections = (result.boxes is not None and len(result.boxes) > 0
                          and result.boxes.id is not None)

        if not has_detections:
            # No people detected at all — keep tracking state, record gap
            frames_lost += 1
            visible_ids = []
            angle_history.append(None)
            boat_avg_history.append(None)
            if frames_lost > MAX_LOST_FRAMES and tracked_id is not None:
                print(f"Rower ID {tracked_id} lost for {frames_lost} frames, will auto-select on return")
                tracked_id = None
            return f

        # Extract track IDs, boxes, keypoints
        track_ids = result.boxes.id.cpu().numpy().astype(int)  # (N,)
        all_boxes = result.boxes.xyxy.cpu().numpy()             # (N, 4)
        has_kp = result.keypoints is not None and len(result.keypoints) > 0
        all_kp = result.keypoints.data.cpu().numpy() if has_kp else None  # (N, 17, 3) or None

        # Sort by bbox area (largest first) for display ordering & auto-select
        areas = (all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1])
        order = np.argsort(-areas)
        visible_ids = track_ids[order].tolist()

        # --- Resolve tracked person ---
        # If we have no tracked_id yet, or it's been lost too long, pick the largest person
        if tracked_id is None or (tracked_id not in track_ids and frames_lost > MAX_LOST_FRAMES):
            tracked_id = visible_ids[0]
            frames_lost = 0
            print(f"Auto-selected rower ID {tracked_id}")

        # Check if our tracked person is in this frame
        if tracked_id in track_ids:
            frames_lost = 0
            primary_det_idx = np.where(track_ids == tracked_id)[0][0]
        else:
            # Tracked person missing this frame — don't switch, just record gap
            frames_lost += 1
            primary_det_idx = None

        # --- Draw all detections ---
        for i in range(len(track_ids)):
            tid = track_ids[i]
            x1, y1, x2, y2 = all_boxes[i].astype(int)
            is_primary = (i == primary_det_idx)

            # Draw skeleton if keypoints available
            if all_kp is not None:
                kp = all_kp[i]
                conf = kp[:, 2]
                draw_skeleton(f, kp, conf, highlight=is_primary)

            # Draw bounding box
            if is_primary:
                cv2.rectangle(f, (x1, y1), (x2, y2), (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(f, f"TRACKED ID:{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(f, (x1, y1), (x2, y2), (80, 80, 80), 1, cv2.LINE_AA)
                cv2.putText(f, f"ID:{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        # --- Compute back angle for tracked rower ---
        if primary_det_idx is not None and all_kp is not None:
            primary_kp = all_kp[primary_det_idx]
            primary_conf = primary_kp[:, 2]
            back = compute_back_angle(primary_kp, primary_conf)
            angle_history.append(back)

            # Draw angle label at the hip joint
            if back is not None:
                if kp_visible(primary_kp, [5, 11, 13], primary_conf):
                    draw_angle_label(f, primary_kp, primary_conf, [5, 11, 13],
                                     back, GRAPH_COLOR)
                elif kp_visible(primary_kp, [6, 12, 14], primary_conf):
                    draw_angle_label(f, primary_kp, primary_conf, [6, 12, 14],
                                     back, GRAPH_COLOR)
        else:
            angle_history.append(None)

        # --- Compute boat average back angle (all rowers except tracked) ---
        if all_kp is not None:
            others = []
            for i in range(len(track_ids)):
                if i == primary_det_idx:
                    continue
                a = compute_back_angle(all_kp[i], all_kp[i][:, 2])
                if a is not None:
                    others.append(a)
            boat_avg_history.append(round(sum(others) / len(others)) if others else None)
        else:
            boat_avg_history.append(None)

        return f

    def seek_to(target_frame):
        nonlocal frame_num, frame, ended
        target_frame = max(0, min(target_frame, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        reset_tracker()
        ret, f = cap.read()
        if ret:
            frame_num = target_frame + 1
            ended = False
            frame = process_frame(f)
        return ret

    while True:
        if not paused and not ended:
            ret, raw = cap.read()
            if not ret:
                ended = True
            else:
                frame_num += 1
                frame = process_frame(raw)

        if ended:
            state = "ended - R to replay"
        elif paused:
            state = "paused"
        else:
            state = "playing"

        if frame is not None:
            # Resize video for display
            disp = cv2.resize(frame, (disp_w, disp_h))
            draw_hud(disp, fps_actual, conf_thresh, state, frame_num,
                     total_frames, fps_video, tracked_id, len(visible_ids))

            # Build graph panel
            graph = draw_graph_panel(disp_w, angle_history, boat_avg_history, fps_video)

            # Combine
            combined = np.vstack([disp, graph])
            cv2.imshow(window_name, combined)

            if writer and not paused and not ended:
                writer.write(combined)

        # Timing
        if not paused and not ended:
            delay = max(1, int(1000 / fps_video) - int(elapsed * 1000))
        else:
            delay = 30
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            if ended:
                angle_history.clear()
                boat_avg_history.clear()
                reset_tracker()
                seek_to(0)
                paused = False
                ended = False
            else:
                paused = not paused
        elif key == ord("r"):
            angle_history.clear()
            boat_avg_history.clear()
            reset_tracker()
            seek_to(0)
            paused = False
            ended = False
        elif key == 81 or key == 2:  # LEFT — step back one frame
            paused = True
            seek_to(max(0, frame_num - 2))
        elif key == 83 or key == 3:  # RIGHT — step forward one frame
            paused = True
            if frame_num < total_frames:
                ret, raw = cap.read()
                if ret:
                    frame_num += 1
                    frame = process_frame(raw)
        elif key == ord("\t"):  # TAB — cycle through visible rower IDs
            if visible_ids:
                if tracked_id in visible_ids:
                    idx = visible_ids.index(tracked_id)
                    tracked_id = visible_ids[(idx + 1) % len(visible_ids)]
                else:
                    tracked_id = visible_ids[0]
                frames_lost = 0
                print(f"Switched to rower ID {tracked_id}")
        elif key == ord("s"):
            if frame is not None:
                path = f"screenshot_{frame_num:05d}.png"
                cv2.imwrite(path, combined)
                print(f"Screenshot saved: {path}")
        elif key in (ord("+"), ord("=")):
            conf_thresh = min(conf_thresh + 0.05, 0.95)
            print(f"Confidence: {conf_thresh:.2f}")
        elif key in (ord("-"), ord("_")):
            conf_thresh = max(conf_thresh - 0.05, 0.05)
            print(f"Confidence: {conf_thresh:.2f}")

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {args.output}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
