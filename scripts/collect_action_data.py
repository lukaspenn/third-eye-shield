#!/usr/bin/env python3
"""
Collect kinematic training data for the 10 selected actions.

Usage:
    DISPLAY=:0 python3 scripts/collect_action_data.py --action 0
    DISPLAY=:0 python3 scripts/collect_action_data.py --action 0 --subject 2

Workflow:
    1. Run the script with --action <id>
    2. Stand in front of the camera (~2m away)
    3. Press SPACE to start recording (30 frames = 1 second)
    4. Perform the action during the red-border recording window
    5. Repeat 20 times per action
    6. Press Q to quit, move to next action

Files saved to: collections/
    S01_A00_001_raw.npy   <- raw skeleton buffer (32, 17, 2)
    S01_A00_001_kin.npy   <- kinematic feature vector (160,)
"""
import os, sys, signal, time, argparse, select, termios, tty
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.kinematics import extract_kinematic_features

# ── The 10 Selected Actions ──────────────────────────────────────────────
ACTIONS = {
    0: "clapping",
    1: "arm circles",
    2: "drink water",
    3: "falling",
    4: "kicking something",
    5: "sit down and up",
    6: "pointing with finger",
    7: "phone call",
    8: "punching/slapping",
    9: "pushing other person",
}

W, H = 800, 480
SEQ_LEN = 32       # frames per clip (~1 second at 30 FPS, matches TCN input)
CONF_THR = 0.25    # joint confidence threshold
MIN_JOINTS = 5     # minimum visible joints to accept a frame

# MoveNet 17-joint bones for skeleton drawing
MOVENET_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

_running = True
def _stop(sig, frame):
    global _running; _running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


class JointSmoother:
    """Lightweight per-joint EMA to reduce MoveNet quantisation jitter.
    Only smooths x, y; confidence is passed through unchanged.
    alpha=1.0 means no smoothing (raw), lower = more smoothing."""
    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha
        self._prev = None

    def __call__(self, kps: np.ndarray) -> np.ndarray:
        """kps: (17, 3)  [x_px, y_px, conf]  ->  smoothed copy"""
        out = kps.copy()
        if self._prev is None:
            self._prev = kps[:, :2].copy()
        else:
            out[:, :2] = self.alpha * kps[:, :2] + (1 - self.alpha) * self._prev
            # Only update prev for joints that are actually visible
            visible = kps[:, 2] > CONF_THR
            self._prev[visible] = out[visible, :2]
        return out

    def reset(self):
        self._prev = None

def draw_skeleton(img, kps):
    """Draw MoveNet 17-joint skeleton. kps: (17,3) = [x_px, y_px, conf]."""
    ih, iw = img.shape[:2]
    def px(i):
        x, y = int(kps[i, 0]), int(kps[i, 1])
        return (max(0, min(x, iw-1)), max(0, min(y, ih-1)))
    for a, b in MOVENET_BONES:
        if kps[a, 2] > CONF_THR and kps[b, 2] > CONF_THR:
            cv2.line(img, px(a), px(b), (0, 0, 0), 5)
            cv2.line(img, px(a), px(b), (255, 255, 255), 2)
    for j in range(17):
        if kps[j, 2] > CONF_THR:
            cv2.circle(img, px(j), 4, (0, 0, 255), -1)

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    parser = argparse.ArgumentParser(description="Collect action training data")
    parser.add_argument('--action', type=int, required=True,
                        help=f"Action ID: {', '.join(f'{k}={v}' for k,v in ACTIONS.items())}")
    parser.add_argument('--subject', type=int, default=1, help="Subject number (default 1)")
    args = parser.parse_args()

    if args.action not in ACTIONS:
        print(f"ERROR: Unknown action ID {args.action}")
        print(f"Valid IDs:")
        for k, v in ACTIONS.items():
            print(f"  {k} = {v}")
        sys.exit(1)

    action_name = ACTIONS[args.action]
    out_dir = Path("collections")
    out_dir.mkdir(exist_ok=True)

    # Count existing samples for this subject + action
    prefix = f"S{args.subject:02d}_A{args.action:02d}"
    existing = sorted(out_dir.glob(f"{prefix}_*_kin.npy"))
    sample_num = len(existing) + 1

    # ── Camera ────────────────────────────────────────────────────────────
    print("[INIT] RealSense...")
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    # Depth not needed for skeleton collection
    pipeline.start(cfg)
    for _ in range(15): pipeline.wait_for_frames()  # warmup

    # ── MoveNet Lightning int8 -- fast single-pose ──
    print("[INIT] MoveNet Lightning int8...")
    from models.movenet_pose_extractor import MoveNetPoseExtractor
    pose = MoveNetPoseExtractor(
        model_path='models/movenet_lightning_int8.tflite',
        conf_threshold=0.25
    )
    smoother = JointSmoother(alpha=0.55)  # mild EMA to reduce jitter

    print()
    print("=" * 60)
    print(f"  ACTION {args.action}: {action_name.upper()}")
    print(f"  Subject: {args.subject}   |   Next sample: #{sample_num}")
    print(f"  ------------------------------------------------")
    print(f"  SPACE  = start 5-second countdown")
    # Set terminal to non-blocking mode for remote keyboard input
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Collect", W, H)

    state = "IDLE"          # IDLE | COUNTDOWN | RECORDING
    skel_buf = []
    countdown_start = 0
    COUNTDOWN_SEC = 5       # 5-second countdown to get into position

    try:
        while _running:
            try:
                frames = pipeline.wait_for_frames(100)
            except KeyboardInterrupt:
                break
            if not frames:
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())

            # Extract skeleton (MoveNet Lightning) + lightweight EMA smoothing
            kps_raw, _, detected = pose.extract(rgb, draw=False)  # (17,3) [x_px, y_px, conf]
            kps = smoother(kps_raw)
            n_j = int(np.sum(kps[:, 2] > CONF_THR))

            # Display with scaled keypoints (convert RGB -> BGR for OpenCV)
            disp = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (W, H))
            sx, sy = W / rgb.shape[1], H / rgb.shape[0]
            kps_disp = kps.copy()
            kps_disp[:, 0] *= sx
            kps_disp[:, 1] *= sy
            if n_j >= MIN_JOINTS:
                draw_skeleton(disp, kps_disp)

            # ── State machine ─────────────────────────────────────────────
            if state == "IDLE":
                cv2.putText(disp, f"[{action_name}] Press SPACE to record #{sample_num}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                if n_j < MIN_JOINTS:
                    cv2.putText(disp, "NO SKELETON - stand in view",
                                (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            elif state == "COUNTDOWN":
                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SEC - elapsed
                
                # Draw big countdown so you can see it from far away
                if remaining > 0:
                    fraction = remaining - int(remaining)
                    # Blink effect on last second
                    if remaining < 1.0 and fraction < 0.5:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 255)
                        
                    cv2.putText(disp, f"{int(remaining)+1}", (W//2 - 50, H//2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 5.0, color, 10)
                    cv2.putText(disp, f"Get Ready: {action_name}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
                else:
                    state = "RECORDING"
                    skel_buf = []
                    print(f"  --> Recording #{sample_num} NOW!")

            elif state == "RECORDING":
                if n_j >= MIN_JOINTS:
                    # Save native MoveNet 17-joint normalised (x, y)
                    h_img, w_img = rgb.shape[:2]
                    xy = np.stack([kps[:, 0] / w_img,
                                   kps[:, 1] / h_img], axis=-1)  # (17, 2)
                    skel_buf.append(xy)

                # Red border = recording
                cv2.rectangle(disp, (0, 0), (W-1, H-1), (0, 0, 255), 12)
                progress = len(skel_buf)
                cv2.putText(disp, f"REC {progress}/{SEQ_LEN}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                # Progress bar
                bar_w = int((progress / SEQ_LEN) * (W - 40))
                cv2.rectangle(disp, (20, H-30), (20 + bar_w, H-10), (0, 0, 255), -1)
                cv2.rectangle(disp, (20, H-30), (W-20, H-10), (255, 255, 255), 1)

                if progress >= SEQ_LEN:
                    # Save raw + kinematic
                    fname = f"{prefix}_{sample_num:03d}"
                    np.save(out_dir / f"{fname}_raw.npy", np.array(skel_buf))

                    kin = extract_kinematic_features(skel_buf)
                    np.save(out_dir / f"{fname}_kin.npy", kin)

                    print(f"  [SAVED] {fname}  |  kin_dim={len(kin)}")
                    sample_num += 1
                    skel_buf = []
                    state = "IDLE"

            # Show frame
            cv2.imshow("Collect", disp)
            key = cv2.waitKey(1) & 0xFF

            # Check for remote SSH keyboard input
            ssh_key = None
            if isData():
                ssh_key = sys.stdin.read(1)

            if key == ord('q') or key == 27 or ssh_key == 'q':
                break
            elif (key == ord(' ') or ssh_key == ' ') and state == "IDLE":
                if n_j >= MIN_JOINTS:
                    state = "COUNTDOWN"
                    countdown_start = time.time()
                else:
                    print("  [!] No skeleton visible, cannot start recording")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        pipeline.stop()
        cv2.destroyAllWindows()
        total = len(list(out_dir.glob(f"{prefix}_*_kin.npy")))
        print(f"\n  Done. Total samples for {action_name}: {total}")

if __name__ == '__main__':
    main()
