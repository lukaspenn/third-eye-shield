#!/usr/bin/env python3
"""
Depth Skeleton + Anomaly Demo  --  800x480
Raw RealSense depth colourised by SDK + MoveNet skeleton overlay + AE score.
Uses MoveNet Lightning for fast skeleton (~47ms vs 168ms Thunder on Pi 4).
Draws raw MoveNet 17-joint skeleton directly --- no stabilizer, no NTU conversion lag.

SSH Keyboard Controls:
    R  = Start / stop video recording
    S  = Save screenshot
    Q  = Quit
"""
import signal, sys, time, pickle, collections, select, termios, tty, os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.process import kill_previous_instances
kill_previous_instances()

import numpy as np
import cv2
import yaml
import tensorflow as tf
import pyrealsense2 as rs

from src.utils.kinematics import extract_kinematic_features
from src.utils.skeleton import MOVENET_BONES, JOINT_CONF_THR, JointSmoother, draw_skeleton
from src.utils.autoencoder import DepthAutoencoder

sys.stdout.reconfigure(line_buffering=True)

_running = True
def _stop(sig, frame):
    global _running; _running = False
signal.signal(signal.SIGINT,  _stop)
signal.signal(signal.SIGTERM, _stop)

W, H = 800, 480


def draw_skeleton_movenet(img, keypoints):
    """Draw raw MoveNet 17-joint skeleton.  keypoints: (17,3) = [x_px, y_px, conf]."""
    ih, iw = img.shape[:2]
    def px(i):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        return (max(0, min(x, iw-1)), max(0, min(y, ih-1)))
    for a, b in MOVENET_BONES:
        if keypoints[a, 2] > JOINT_CONF_THR and keypoints[b, 2] > JOINT_CONF_THR:
            cv2.line(img, px(a), px(b), (0, 0, 0),        7)
            cv2.line(img, px(a), px(b), (255, 255, 255),   3)
    for j in range(17):
        if keypoints[j, 2] > JOINT_CONF_THR:
            p = px(j)
            r = 8 if j == 0 else 5
            cv2.circle(img, p, r+2, (0, 0, 0),    -1)
            cv2.circle(img, p, r,   (0, 0, 255),   -1)

# Kept for backward compatibility; new code uses DepthAutoencoder from shared module
class AE(DepthAutoencoder):
    """Legacy alias for DepthAutoencoder."""
    def __init__(self, path, size=(64, 64)):
        super().__init__(path, size)

def main():
    from models.movenet_pose_extractor import MoveNetPoseExtractor

    print("="*60)
    print("  Depth Skeleton + Anomaly Demo  |  800x480")
    print("  SSH Keyboard: R=Record  S=Screenshot  Q=Quit")
    print("="*60)

    # --- MoveNet Lightning int8 + DepthROI -- fast single-pose, depth-gated ---
    print("[INIT] MoveNet Lightning int8 + DepthROI...")
    pose = MoveNetPoseExtractor(
        model_path='models/movenet_lightning_int8.tflite',
        conf_threshold=0.25
    )
    smoother = JointSmoother(alpha=0.55)  # mild EMA to reduce jitter

    ae, ae_thr = None, 0.0912  # anomaly when mse < threshold (person present = low MSE)
    try:
        root = Path(__file__).resolve().parents[1]
        cfg = {}
        cp = root/'config.yaml'
        if cp.exists():
            cfg = yaml.safe_load(cp.read_text()) or {}
        s1 = cfg.get('stage1', {})
        ae_thr = float(s1.get('reconstruction_threshold', ae_thr))
        mp = root / s1.get('model_path', 'models/depth_autoencoder_full.tflite')
        sz = tuple(s1.get('input_size', [64, 64]))
        if mp.exists():
            print(f"[INIT] AE: {mp.name}  threshold={ae_thr:.4f}")
            ae = AE(mp, sz)
    except Exception as e:
        print(f"[INIT] AE disabled: {e}")

    # --- Action Classifier (RandomForest) --------------------------------
    action_model = None
    action_classes = {}
    action_label = ""
    action_conf  = 0.0
    ACTION_CONF_THR = 0.38       # only accept predictions above this confidence
    ACTION_VOTE_N   = 2          # require N consecutive same predictions
    THREAT_VOTE_N   = 3          # extra votes required for threat actions
    FALLING_CONF_THR = 0.55      # higher bar for 'falling' to reduce false positives
    action_vote_buf = []         # recent predictions for temporal voting
    ACTION_HOLD_SEC = 2.0        # keep last action label for this long before idle
    action_last_t   = 0.0        # timestamp of last confident prediction
    SEQ_LEN = 30  # 1 second buffer at 30 FPS
    skel_buf = collections.deque(maxlen=SEQ_LEN)
    prev_n_j = 0                 # joint count from previous frame (dropout detection)
    rf_path = root / 'models' / 'action_rf.pkl'
    if rf_path.exists():
        try:
            blob = pickle.load(open(rf_path, 'rb'))
            action_model = blob['model']
            action_classes = blob['classes']
            print(f"[INIT] Action RF: {blob['n_classes']} classes, {blob['n_samples']} training samples")
        except Exception as e:
            print(f"[INIT] Action RF disabled: {e}")
    else:
        print(f"[INIT] Action RF: no model at {rf_path.name} (skipping)")

    print("[INIT] RealSense...")
    pipeline  = rs.pipeline()
    rscfg     = rs.config()
    rscfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    rscfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  30)
    pipeline.start(rscfg)
    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()          # SDK built-in depth coloriser
    colorizer.set_option(rs.option.color_scheme, 0)   # 0=Jet, 2=WhiteToBlack, 9=Hue

    # Denoise pipeline (all C++, fast)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude,    2)    # passes (1-5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # blend strength
    spatial.set_option(rs.option.filter_smooth_delta, 20)   # edge threshold (mm)

    for _ in range(15):
        pipeline.wait_for_frames()
    print("[INIT] Ready.\n")

    # --- SSH keyboard setup ------------------------------------------------
    _has_tty = False
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        _has_tty = True
    except (termios.error, ValueError, OSError):
        print("[WARN] No interactive terminal - SSH keys disabled (use cv2 window for Q)")
    def ssh_key():
        if not _has_tty:
            return None
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return sys.stdin.read(1)
        except Exception:
            pass
        return None

    # --- Recording / screenshot dirs ---------------------------------------
    clips_dir = Path('clips')
    clips_dir.mkdir(exist_ok=True)
    screenshots_dir = Path('screenshots')
    screenshots_dir.mkdir(exist_ok=True)

    # --- Alert log ---------------------------------------------------------
    alert_log_dir = Path('logs/alerts')
    alert_log_dir.mkdir(parents=True, exist_ok=True)
    alert_log_path = alert_log_dir / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    alert_log_f = open(alert_log_path, 'w')
    alert_log_f.write('timestamp,event,action,confidence,ae_mse\n')
    ALERT_COOLDOWN = 5.0   # seconds between repeated alerts
    last_alert_t   = 0.0
    THREAT_ACTIONS  = {'punching/slapping', 'pushing other person', 'kicking something', 'falling'}

    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Demo', W, H)

    fps_avg = 25.0
    ae_mse  = None
    frame_n = 0
    t_prev  = time.time()

    # Video writer state
    recording   = False
    video_writer = None
    rec_start   = 0.0
    rec_path    = None

    try:
      while _running:
        try:
            frames = pipeline.wait_for_frames(100)  # 100 ms → Ctrl+C fires within 1 frame
        except KeyboardInterrupt:
            break
        if not frames:
            continue
        aligned = align.process(frames)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            continue

        frame_n += 1
        rgb = np.asanyarray(cf.get_data())   # (480,848,3) RGB

        # --- Depth: spatial filter then colourise -------------------------
        df_f        = spatial.process(df)
        depth_color = np.asanyarray(colorizer.colorize(df_f).get_data())  # RGB
        depth_bgr   = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
        disp        = cv2.resize(depth_bgr, (W, H))

        raw = np.asanyarray(df.get_data())   # original for AE

        # --- AE first (every 3 frames, cached) - gates skeleton below ------
        # NOTE: AE threshold logic is intentionally inverted:
        #   low MSE  = AE reconstructs well = person present = ANOMALY (trigger)
        #   high MSE = empty/noisy scene     = NORMAL
        if ae and frame_n % 3 == 0:
            try:
                ae_mse = ae.mse(raw)
                if frame_n % 30 == 0:
                    print(f"[AE] mse={ae_mse:.5f}  thr={ae_thr:.5f}  {'ANOMALY' if ae_mse < ae_thr else 'normal'}")
            except Exception as ex:
                print(f"[AE] error: {ex}")
                ae_mse = None
        anom = ae_mse is not None and ae_mse < ae_thr

        # --- Skeleton + Action ONLY when anomaly (person detected) ---------
        n_j = 0
        if anom:
            kps_raw, _, detected = pose.extract(rgb, draw=False, depth_frame=raw)
            kps = smoother(kps_raw)
            sx, sy = W / rgb.shape[1], H / rgb.shape[0]
            kps_disp = kps.copy()
            kps_disp[:, 0] *= sx
            kps_disp[:, 1] *= sy
            n_j = int(np.sum(kps[:, 2] > JOINT_CONF_THR))
            if n_j >= 5:
                draw_skeleton_movenet(disp, kps_disp)

            # Joint-stability gate: skip action when joints are poor or dropping fast
            joint_stable = (n_j >= 5 and not (prev_n_j >= 5 and n_j < prev_n_j * 0.5))
            prev_n_j = n_j

            if action_model and joint_stable:
                h_img, w_img = rgb.shape[:2]
                xy = np.stack([kps[:, 0] / w_img,
                               kps[:, 1] / h_img], axis=-1)
                skel_buf.append(xy)
                if len(skel_buf) == SEQ_LEN and frame_n % 10 == 0:
                    try:
                        feats = extract_kinematic_features(list(skel_buf))
                        probs = action_model.predict_proba(feats.reshape(1, -1))[0]
                        pred_idx = np.argmax(probs)
                        pred_class = action_model.classes_[pred_idx]
                        pred_conf  = probs[pred_idx]
                        pred_name = action_classes.get(pred_class, f"class_{pred_class}")
                        # Higher confidence bar for 'falling'
                        eff_thr = FALLING_CONF_THR if pred_name == 'falling' else ACTION_CONF_THR
                        if pred_conf >= eff_thr:
                            action_vote_buf.append(pred_name)
                            need = THREAT_VOTE_N if pred_name in THREAT_ACTIONS else ACTION_VOTE_N
                            if len(action_vote_buf) > need:
                                action_vote_buf.pop(0)
                            if (len(action_vote_buf) >= need and
                                    len(set(action_vote_buf)) == 1):
                                action_label = pred_name
                                action_conf  = pred_conf
                                action_last_t = time.time()
                        else:
                            action_vote_buf.clear()
                            # Hold last label for ACTION_HOLD_SEC before going idle
                            if action_label and action_label != "(idle)" and (time.time() - action_last_t) < ACTION_HOLD_SEC:
                                pass  # keep current label
                            else:
                                action_label = "(idle)"
                                action_conf = 0.0
                    except Exception as ex:
                        print(f"[ACTION] error: {ex}")
        else:
            # No anomaly -- clear action state, reset smoother & tracking
            skel_buf.clear()
            action_vote_buf.clear()
            action_label = ""
            action_conf = 0.0
            smoother.reset()
            pose.reset_tracking()

        # --- FPS EMA --------------------------------------------------------
        now     = time.time()
        fps_avg = 0.9*fps_avg + 0.1*(1.0/max(now-t_prev, 1e-6))
        t_prev  = now

        # --- HUD ------------------------------------------------------------
        lock_tag = "  [LOCKED]" if pose.is_locked else ""
        cv2.putText(disp, f"FPS {fps_avg:.0f}  Joints {n_j}/17{lock_tag}",
                    (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,0),  3)
        cv2.putText(disp, f"FPS {fps_avg:.0f}  Joints {n_j}/17{lock_tag}",
                    (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if pose.is_locked else (255,255,255), 2)

        # --- Action label on HUD -------------------------------------------
        if anom:
            disp_label = action_label if action_label else "(idle)"
            is_threat = disp_label in THREAT_ACTIONS
            if is_threat:
                action_col = (0, 0, 255)   # red for threat actions
            elif disp_label == "(idle)":
                action_col = (160, 160, 160)  # gray for idle
            else:
                action_col = (0, 255, 255)  # yellow for normal actions
            conf_str = f" ({action_conf:.0%})" if action_conf > 0 else ""
            atxt = f"Action: {disp_label}{conf_str}"
            cv2.putText(disp, atxt, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(disp, atxt, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_col, 2)

            # Threat line under action
            if is_threat and action_conf >= 0.45:
                cv2.putText(disp, "THREAT", (8, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(disp, "THREAT", (8, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Log alert (with cooldown)
                if now - last_alert_t > ALERT_COOLDOWN:
                    last_alert_t = now
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    alert_log_f.write(f"{ts},THREAT,{action_label},{action_conf:.3f},{ae_mse:.5f}\n")
                    alert_log_f.flush()
                    print(f"  [ALERT] THREAT: {action_label} ({action_conf:.0%})  MSE={ae_mse:.4f}")

        # Log anomaly entrance (person appears)
        if anom and not getattr(main, '_was_anom', False):
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            alert_log_f.write(f"{ts},PERSON_ENTERED,,, {ae_mse:.5f}\n")
            alert_log_f.flush()
            print(f"  [ALERT] Person entered (MSE={ae_mse:.4f})")
        if not anom and getattr(main, '_was_anom', False):
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            mse_str = f"{ae_mse:.5f}" if ae_mse is not None else "0"
            alert_log_f.write(f"{ts},PERSON_LEFT,,,{mse_str}\n")
            alert_log_f.flush()
            print(f"  [ALERT] Person left")
        main._was_anom = anom

        # Green border = normal, Red border = anomaly (always visible)
        border_col = (0, 0, 255) if anom else (0, 220, 0)
        cv2.rectangle(disp, (0,0), (W-1,H-1), border_col, 6)

        if ae_mse is not None:
            col   = (0,0,255) if anom else (0,200,80)
            label = "!! ANOMALY !!" if anom else "normal"
            cv2.putText(disp, f"AE {label}  {ae_mse:.4f}",
                        (8,H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0),   3)
            cv2.putText(disp, f"AE {label}  {ae_mse:.4f}",
                        (8,H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col,        2)

        # --- Recording indicator on HUD ------------------------------------
        if recording:
            elapsed = time.time() - rec_start
            mins, secs = int(elapsed // 60), int(elapsed % 60)
            # Blinking red circle
            if int(elapsed * 2) % 2 == 0:
                cv2.circle(disp, (W - 30, 25), 10, (0, 0, 255), -1)
            cv2.putText(disp, f"REC {mins:02d}:{secs:02d}", (W - 150, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(disp, f"REC {mins:02d}:{secs:02d}", (W - 150, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            # Write frame
            if video_writer is not None:
                video_writer.write(disp)

        cv2.imshow('Demo', disp)
        wk = cv2.waitKey(1) & 0xFF
        if wk in (ord('q'), 27):
            break

        # --- SSH keyboard --------------------------------------------------
        k = ssh_key()
        if k is not None:
            if k in ('r', 'R'):
                if not recording:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    rec_path = clips_dir / f"clip_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(rec_path), fourcc, 15.0, (W, H))
                    rec_start = time.time()
                    recording = True
                    print(f"  [REC] Recording started: {rec_path.name}")
                else:
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    elapsed = time.time() - rec_start
                    print(f"  [REC] Stopped ({elapsed:.1f}s) -> {rec_path.name}")
            elif k in ('s', 'S'):
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                sp = screenshots_dir / f"screenshot_{ts}.png"
                cv2.imwrite(str(sp), disp)
                print(f"  [SCREENSHOT] {sp.name}")
            elif k in ('l', 'L'):
                if pose.is_locked:
                    pose.unlock()
                    print("  [LOCK] Unlocked - tracking nearest person")
                else:
                    pose.lock_on(depth_frame=raw)
                    print("  [LOCK] Locked onto current target")
            elif k in ('q', 'Q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if old_settings:
            try: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception: pass
        if video_writer:
            video_writer.release()
            print(f"  [REC] Saved: {rec_path.name}")
        alert_log_f.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        total_clips = len(list(clips_dir.glob('*.mp4')))
        total_shots = len(list(screenshots_dir.glob('*.png')))
        print(f"[INFO] Stopped.  Clips: {total_clips}  Screenshots: {total_shots}")
        print(f"[INFO] Alert log: {alert_log_path}")

if __name__ == '__main__':
    main()
