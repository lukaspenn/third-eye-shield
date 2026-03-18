#!/usr/bin/env python3
"""
Science Fair Competition Showcase - Guided Demo Script

This script runs the full pipeline with a GUIDED SCENARIO mode designed to
impress science fair judges. It walks through each capability of the system
with on-screen prompts and auto-records each demo segment as a short clip.

Usage (from project root, over SSH):
    DISPLAY=:0 python3 scripts/science_fair_showcase.py

SSH Keyboard Controls:
    SPACE / N  = Next scenario step
    R          = Toggle manual recording
    S          = Screenshot
    F          = Free-run mode (normal demo, no prompts)
    B          = Back to guided mode
    Q          = Quit

Demo Scenario Flow (press SPACE to advance):
  1. Empty room - Green border, "Normal" - shows AE baseline
  2. Person enters - Red border, "Anomaly" - AE detects presence
  3. Safe actions  - Skeleton overlay + "clapping" / "arm circles"
  4. Threat actions - Red banner "THREAT DETECTED" + "punching" / "kicking"
  5. Person leaves  - Back to green, AE returns to normal
  6. Privacy demo   - Shows depth-only (no RGB ever stored)

Each step auto-records a 5-second clip to clips/ for later playback.
"""
import os, sys, signal, time, pickle, collections, select, termios, tty
from pathlib import Path
from datetime import datetime

# Kill any previous instance that holds the RealSense camera
def _kill_previous():
    my_pid = os.getpid()
    scripts = ('touchscreen_launcher.py', 'rule_based_demo_depth_overlay.py',
               'science_fair_showcase.py')
    import subprocess
    try:
        out = subprocess.check_output(['ps', 'aux'], text=True)
        for line in out.splitlines():
            if not any(s in line for s in scripts):
                continue
            parts = line.split()
            pid = int(parts[1])
            if pid == my_pid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"[INIT] Killed previous instance (PID {pid})")
                time.sleep(0.5)
            except ProcessLookupError:
                pass
    except Exception:
        pass

_kill_previous()

import numpy as np
import cv2
import yaml
import tensorflow as tf
import pyrealsense2 as rs

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.kinematics import extract_kinematic_features

_running = True
def _stop(sig, frame):
    global _running; _running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

W, H = 800, 480
JOINT_CONF_THR = 0.25
MOVENET_BONES = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]
THREAT_ACTIONS = {'punching/slapping', 'pushing other person', 'kicking something', 'falling'}

# ── Scenario steps ────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "title": "Stage 1: Empty Room Baseline",
        "prompt": "Make sure NO ONE is in frame.\nThis shows the depth autoencoder learning the 'normal' empty scene.",
        "prompt_short": "Clear the scene - show NORMAL state",
        "auto_clip_sec": 5,
    },
    {
        "title": "Stage 2: Person Enters - Anomaly Detection",
        "prompt": "Walk into the camera's view.\nThe border turns RED and the AE triggers ANOMALY.",
        "prompt_short": "Walk into frame - show ANOMALY trigger",
        "auto_clip_sec": 5,
    },
    {
        "title": "Stage 3: Safe Action - Skeleton + Classification",
        "prompt": "Perform a SAFE action: clapping, arm circles, or drink water.\nSkeleton overlay + action label will appear.",
        "prompt_short": "Do a safe action (clapping, arm circles...)",
        "auto_clip_sec": 8,
    },
    {
        "title": "Stage 4: Threat Action - Alert System",
        "prompt": "Perform a THREAT action: punching, kicking, or pushing.\nRed THREAT DETECTED banner will appear.",
        "prompt_short": "Do a threat action (punch, kick, push...)",
        "auto_clip_sec": 8,
    },
    {
        "title": "Stage 5: Person Leaves - Return to Normal",
        "prompt": "Walk OUT of the camera's view.\nBorder returns to GREEN - system resets to normal.",
        "prompt_short": "Leave the frame - show return to NORMAL",
        "auto_clip_sec": 5,
    },
    {
        "title": "Stage 6: Privacy-First Design",
        "prompt": "Notice: only DEPTH data is shown - no RGB/color image is ever stored.\nThis preserves privacy while still detecting threats.",
        "prompt_short": "Depth-only = privacy preserved",
        "auto_clip_sec": 5,
    },
]

class JointSmoother:
    def __init__(self, alpha=0.55):
        self.alpha, self._prev = alpha, None
    def __call__(self, kps):
        out = kps.copy()
        if self._prev is None:
            self._prev = kps[:,:2].copy()
        else:
            out[:,:2] = self.alpha*kps[:,:2] + (1-self.alpha)*self._prev
            vis = kps[:,2] > JOINT_CONF_THR
            self._prev[vis] = out[vis,:2]
        return out
    def reset(self):
        self._prev = None

def draw_skeleton(img, kps):
    ih, iw = img.shape[:2]
    def px(i):
        x, y = int(kps[i,0]), int(kps[i,1])
        return (max(0,min(x,iw-1)), max(0,min(y,ih-1)))
    for a,b in MOVENET_BONES:
        if kps[a,2]>JOINT_CONF_THR and kps[b,2]>JOINT_CONF_THR:
            cv2.line(img, px(a), px(b), (0,0,0), 7)
            cv2.line(img, px(a), px(b), (255,255,255), 3)
    for j in range(17):
        if kps[j,2]>JOINT_CONF_THR:
            p = px(j)
            r = 8 if j==0 else 5
            cv2.circle(img, p, r+2, (0,0,0), -1)
            cv2.circle(img, p, r, (0,0,255), -1)

class AE:
    def __init__(self, path, size=(64,64)):
        self.size = size
        it = tf.lite.Interpreter(model_path=str(path)); it.allocate_tensors()
        self.it, self.inp, self.out = it, it.get_input_details()[0]['index'], it.get_output_details()[0]['index']
    def mse(self, raw):
        d = cv2.resize(raw, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        d = np.clip((d-100)/9900.0, 0, 1)[None,...,None]
        self.it.set_tensor(self.inp, d); self.it.invoke()
        rec = self.it.get_tensor(self.out)[0,...,0]
        return float(np.mean((d[0,...,0]-rec)**2))

def ssh_key():
    try:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
    except Exception:
        pass
    return None

def draw_prompt_box(disp, title, prompt, step, total):
    """Draw a semi-transparent prompt overlay at the bottom."""
    overlay = disp.copy()
    box_h = 120
    cv2.rectangle(overlay, (0, H-box_h), (W, H), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, disp, 0.15, 0, disp)
    # Title
    cv2.putText(disp, f"[{step+1}/{total}] {title}", (12, H-box_h+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    # Prompt lines
    for i, line in enumerate(prompt.split('\n')):
        cv2.putText(disp, line, (12, H-box_h+55 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Navigation hint
    cv2.putText(disp, "SPACE=Next  F=Free-run  Q=Quit", (12, H-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def main():
    from models.movenet_pose_extractor import MoveNetPoseExtractor

    clips_dir = Path('clips')
    clips_dir.mkdir(exist_ok=True)
    screenshots_dir = Path('screenshots')
    screenshots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  SCIENCE FAIR SHOWCASE - Guided Demo")
    print("=" * 60)
    print("  SPACE/N = Next step   F = Free-run   Q = Quit")
    print("  R = Record   S = Screenshot")
    print("=" * 60)

    # --- Models ---
    print("[INIT] MoveNet Lightning int8 + DepthROI...")
    pose = MoveNetPoseExtractor(model_path='models/movenet_lightning_int8.tflite', conf_threshold=0.25)
    smoother = JointSmoother(alpha=0.55)

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(open(root/'config.yaml')) or {}
    s1 = cfg.get('stage1', {})
    ae_thr = float(s1.get('reconstruction_threshold', 0.0912))
    ae = AE(root/s1.get('model_path','models/depth_autoencoder_full.tflite'),
            tuple(s1.get('input_size',[64,64])))
    print(f"[INIT] AE: threshold={ae_thr:.4f}")

    action_model, action_classes = None, {}
    action_label, action_conf = "", 0.0
    ACTION_CONF_THR, ACTION_VOTE_N = 0.38, 2
    THREAT_VOTE_N = 3              # extra votes required for threat actions
    FALLING_CONF_THR = 0.55        # higher bar for 'falling'
    action_vote_buf = []
    ACTION_HOLD_SEC = 2.0        # keep last action label for this long before idle
    action_last_t   = 0.0        # timestamp of last confident prediction
    SEQ_LEN = 30
    skel_buf = collections.deque(maxlen=SEQ_LEN)
    prev_n_j = 0
    rf_path = root/'models'/'action_rf.pkl'
    if rf_path.exists():
        blob = pickle.load(open(rf_path,'rb'))
        action_model, action_classes = blob['model'], blob['classes']
        print(f"[INIT] Action RF: {blob['n_classes']} classes")

    print("[INIT] RealSense...")
    pipeline = rs.pipeline()
    rscfg = rs.config()
    rscfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    rscfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    pipeline.start(rscfg)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    for _ in range(15):
        pipeline.wait_for_frames()
    print("[INIT] Ready.\n")

    _has_tty = False
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        _has_tty = True
    except (termios.error, ValueError, OSError):
        print("[WARN] No interactive terminal - SSH keys disabled")

    cv2.namedWindow('Showcase', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Showcase', W, H)

    # State
    frame_n = 0
    ae_mse = None
    anom = False
    n_j = 0
    fps_avg = 20.0
    t_prev = time.time()

    guided_mode = True
    scenario_idx = 0
    auto_recording = False
    auto_rec_start = 0.0
    auto_writer = None

    manual_recording = False
    manual_writer = None
    manual_rec_start = 0.0
    manual_rec_path = None

    try:
        while _running:
            frames = pipeline.wait_for_frames(200)
            aligned = align.process(frames)
            cf, df = aligned.get_color_frame(), aligned.get_depth_frame()
            if not cf or not df:
                continue
            frame_n += 1
            rgb = np.asanyarray(cf.get_data())
            raw = np.asanyarray(df.get_data())

            df_f = spatial.process(df)
            depth_color = np.asanyarray(colorizer.colorize(df_f).get_data())
            disp = cv2.resize(cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR), (W, H))

            # AE first
            if frame_n % 3 == 0:
                ae_mse = ae.mse(raw)
            anom = ae_mse is not None and ae_mse < ae_thr

            # Skeleton + action only on anomaly
            n_j = 0
            if anom:
                kps_raw, _, _ = pose.extract(rgb, draw=False, depth_frame=raw)
                kps = smoother(kps_raw)
                sx, sy = W/rgb.shape[1], H/rgb.shape[0]
                kps_d = kps.copy(); kps_d[:,0]*=sx; kps_d[:,1]*=sy
                n_j = int(np.sum(kps[:,2]>JOINT_CONF_THR))
                if n_j >= 5:
                    draw_skeleton(disp, kps_d)
                # Joint-stability gate: skip action when joints are poor or dropping fast
                joint_stable = (n_j >= 5 and not (prev_n_j >= 5 and n_j < prev_n_j * 0.5))
                prev_n_j = n_j
                if action_model and joint_stable:
                    h_img, w_img = rgb.shape[:2]
                    xy = np.stack([kps[:,0]/w_img, kps[:,1]/h_img], axis=-1)
                    skel_buf.append(xy)
                    if len(skel_buf)==SEQ_LEN and frame_n%10==0:
                        try:
                            feats = extract_kinematic_features(list(skel_buf))
                            probs = action_model.predict_proba(feats.reshape(1,-1))[0]
                            pi = np.argmax(probs)
                            pc, pconf = action_model.classes_[pi], probs[pi]
                            pn = action_classes.get(pc, f"class_{pc}")
                            eff_thr = FALLING_CONF_THR if pn == 'falling' else ACTION_CONF_THR
                            if pconf >= eff_thr:
                                action_vote_buf.append(pn)
                                need = THREAT_VOTE_N if pn in THREAT_ACTIONS else ACTION_VOTE_N
                                if len(action_vote_buf)>need: action_vote_buf.pop(0)
                                if len(action_vote_buf)>=need and len(set(action_vote_buf))==1:
                                    action_label, action_conf = pn, pconf
                                    action_last_t = time.time()
                            else:
                                action_vote_buf.clear()
                                if action_label and action_label != "(idle)" and (time.time() - action_last_t) < ACTION_HOLD_SEC:
                                    pass  # keep current label
                                else:
                                    action_label, action_conf = "(idle)", 0.0
                        except Exception:
                            pass
            else:
                skel_buf.clear(); action_vote_buf.clear()
                action_label, action_conf = "", 0.0
                smoother.reset()
                pose.reset_tracking()

            # FPS
            now = time.time()
            fps_avg = 0.9*fps_avg + 0.1*(1.0/max(now-t_prev,1e-6))
            t_prev = now

            # --- HUD ---
            border_col = (0,0,255) if anom else (0,220,0)
            cv2.rectangle(disp, (0,0), (W-1,H-1), border_col, 6)

            lock_tag = "  [LOCKED]" if pose.is_locked else ""
            cv2.putText(disp, f"FPS {fps_avg:.0f}  Joints {n_j}/17{lock_tag}",
                        (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(disp, f"FPS {fps_avg:.0f}  Joints {n_j}/17{lock_tag}",
                        (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if pose.is_locked else (255,255,255), 2)

            if action_label:
                is_threat = action_label in THREAT_ACTIONS
                ac = (0,0,255) if is_threat else (0,255,255)
                atxt = f"Action: {action_label} ({action_conf:.0%})"
                cv2.putText(disp, atxt, (8,56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(disp, atxt, (8,56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac, 2)
                if is_threat and action_conf >= 0.45:
                    cv2.putText(disp, "!! THREAT DETECTED !!", (W//2-180, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
                    cv2.putText(disp, "!! THREAT DETECTED !!", (W//2-180, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            if ae_mse is not None:
                col = (0,0,255) if anom else (0,200,80)
                lbl = "!! ANOMALY !!" if anom else "normal"
                y_ae = H - 130 if guided_mode else H - 12
                cv2.putText(disp, f"AE {lbl}  {ae_mse:.4f}", (8,y_ae),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(disp, f"AE {lbl}  {ae_mse:.4f}", (8,y_ae),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            # --- Guided prompt overlay ----------------------------------------
            if guided_mode and scenario_idx < len(SCENARIOS):
                sc = SCENARIOS[scenario_idx]
                draw_prompt_box(disp, sc['title'], sc['prompt'], scenario_idx, len(SCENARIOS))

            # --- Auto-recording for guided scenario ---------------------------
            if auto_recording:
                elapsed = time.time() - auto_rec_start
                sc = SCENARIOS[min(scenario_idx, len(SCENARIOS)-1)]
                limit = sc.get('auto_clip_sec', 5)
                if elapsed < limit:
                    if auto_writer:
                        auto_writer.write(disp)
                    # Recording indicator
                    secs_left = int(limit - elapsed)
                    cv2.circle(disp, (W-30, 60), 8, (0,0,255), -1)
                    cv2.putText(disp, f"AUTO-REC {secs_left}s", (W-170, 67),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
                else:
                    auto_recording = False
                    if auto_writer:
                        auto_writer.release(); auto_writer = None
                    print(f"  [AUTO-REC] Saved clip for step {scenario_idx+1}")

            # Manual recording indicator
            if manual_recording:
                elapsed = time.time() - manual_rec_start
                if int(elapsed*2)%2==0:
                    cv2.circle(disp, (W-30, 25), 10, (0,0,255), -1)
                mins, secs = int(elapsed//60), int(elapsed%60)
                cv2.putText(disp, f"REC {mins:02d}:{secs:02d}", (W-150, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
                if manual_writer:
                    manual_writer.write(disp)

            cv2.imshow('Showcase', disp)
            cv2.waitKey(1)

            # --- SSH keyboard -------------------------------------------------
            k = ssh_key()
            if k is None:
                continue

            if k in (' ', 'n', 'N'):
                # Next scenario step
                if guided_mode and scenario_idx < len(SCENARIOS):
                    # Start auto-recording this step
                    sc = SCENARIOS[scenario_idx]
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    tag = sc['title'].lower().replace(' ','_').replace(':','').replace('-','')[:30]
                    rp = clips_dir / f"showcase_{scenario_idx+1}_{tag}_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    auto_writer = cv2.VideoWriter(str(rp), fourcc, 15.0, (W, H))
                    auto_rec_start = time.time()
                    auto_recording = True
                    print(f"\n  >> Step {scenario_idx+1}/{len(SCENARIOS)}: {sc['title']}")
                    print(f"     {sc['prompt_short']}")
                    print(f"     Auto-recording {sc['auto_clip_sec']}s -> {rp.name}")
                    scenario_idx += 1
                elif guided_mode:
                    print("\n  All scenarios done! Press F for free-run or Q to quit.")
                    guided_mode = False

            elif k in ('f', 'F'):
                guided_mode = False
                if auto_recording and auto_writer:
                    auto_writer.release(); auto_writer = None
                auto_recording = False
                print("  [MODE] Free-run (no prompts)")

            elif k in ('b', 'B'):
                guided_mode = True
                scenario_idx = 0
                print("  [MODE] Guided mode restarted from step 1")

            elif k in ('r', 'R'):
                if not manual_recording:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    manual_rec_path = clips_dir / f"manual_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    manual_writer = cv2.VideoWriter(str(manual_rec_path), fourcc, 15.0, (W, H))
                    manual_rec_start = time.time()
                    manual_recording = True
                    print(f"  [REC] Manual recording -> {manual_rec_path.name}")
                else:
                    manual_recording = False
                    if manual_writer: manual_writer.release(); manual_writer = None
                    print(f"  [REC] Stopped -> {manual_rec_path.name}")

            elif k in ('s', 'S'):
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                sp = screenshots_dir / f"showcase_{ts}.png"
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

    finally:
        if old_settings:
            try: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception: pass
        for w in [auto_writer, manual_writer]:
            if w: w.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        n_clips = len(list(clips_dir.glob('*.mp4')))
        n_shots = len(list(screenshots_dir.glob('*.png')))
        print(f"\n[DONE] {n_clips} clips in clips/  |  {n_shots} screenshots in screenshots/")


if __name__ == '__main__':
    main()
