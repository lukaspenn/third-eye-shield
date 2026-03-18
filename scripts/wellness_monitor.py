#!/usr/bin/env python3
"""
Third Eye Shield — Privacy-Preserving Elderly Wellness Monitor
=======================================================

Multimodal wellness monitoring using depth-first pipeline:
  Stage 1: Depth autoencoder anomaly detection (95% of frames end here)
  Stage 2: MoveNet skeleton pose extraction (only on anomaly)
  Stage 3: Action classification + posture scoring + optional emotion

Wellness Levels:
  0 = Active    (exercising)         → green
  1 = Normal    (daily activity)     → cyan
  2 = Sedentary (inactive >30 min)   → orange
  3 = Concern   (posture/emotion)    → dark orange
  4 = Alert     (fall detected)      → red

Privacy Model:
  - Depth-only continuous monitoring (non-identifiable)
  - RGB accessed locally for pose only when person detected
  - Emotion detection OFF by default (opt-in via --enable-emotion or E key)
  - Face crops processed in-memory only, never stored

SSH Keyboard Controls:
    R  = Start / stop video recording
    S  = Save screenshot
    E  = Toggle emotion detection on/off
    L  = Lock/unlock person tracking
    Q  = Quit
"""
import signal, sys, time, pickle, collections, select, termios, tty, os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.process import kill_previous_instances
kill_previous_instances()

import numpy as np
try:
    import cv2
    _CV2_IMPORT_ERROR = None
except Exception as _e:
    cv2 = None
    _CV2_IMPORT_ERROR = _e
try:
    import yaml
    _YAML_IMPORT_ERROR = None
except Exception as _e:
    yaml = None
    _YAML_IMPORT_ERROR = _e
try:
    import tensorflow as tf
    _TF_IMPORT_ERROR = None
except Exception as _e:
    tf = None
    _TF_IMPORT_ERROR = _e
try:
    import pyrealsense2 as rs
    _RS_IMPORT_ERROR = None
except Exception as _e:
    rs = None
    _RS_IMPORT_ERROR = _e

from src.utils.kinematics import extract_kinematic_features
from src.utils.wellness_features import (
    PostureTracker, SedentaryTracker, compute_wellness_level,
    WELLNESS_NAMES, WELLNESS_COLORS_BGR, ACTIVE_ACTIONS, ALERT_ACTIONS,
    WELLNESS_ACTIVE, WELLNESS_NORMAL, WELLNESS_SEDENTARY, WELLNESS_CONCERN, WELLNESS_ALERT,
)
from src.utils.skeleton import MOVENET_BONES, JOINT_CONF_THR, JointSmoother, draw_skeleton
from src.utils.autoencoder import DepthAutoencoder

sys.stdout.reconfigure(line_buffering=True)

_running = True
def _stop(sig, frame):
    global _running; _running = False
signal.signal(signal.SIGINT,  _stop)
signal.signal(signal.SIGTERM, _stop)

W, H = 800, 480


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Third Eye Shield Wellness Monitor")
    parser.add_argument('--enable-emotion', action='store_true',
                        help='Enable opt-in facial emotion detection (off by default)')
    parser.add_argument('--emotion-model', type=str, default=None,
                        help='Path to TFLite emotion classifier model')
    parser.add_argument('--emotion-features-model', type=str, default=None,
                        help='Path to TFLite feature extractor for few-shot')
    parser.add_argument('--emotion-profile', type=str, default=None,
                        help='User ID for few-shot emotion profile (e.g. UNCLE_TAN)')
    parser.add_argument('--emotion-calibration', type=str, default=None,
                        help='(Legacy) Path to per-user calibration .npy file')
    parser.add_argument('--sedentary-minutes', type=int, default=30,
                        help='Sedentary alert threshold in minutes (default: 30)')
    parser.add_argument('--llm-endpoint', type=str, default=None,
                        help='HTTP endpoint for LLM companion (e.g. http://laptop:5000/chat)')
    parser.add_argument('--telegram-token', type=str, default=None,
                        help='Telegram bot token for family notifications')
    parser.add_argument('--telegram-chat-ids', type=str, nargs='+', default=None,
                        help='Telegram chat IDs to notify (space-separated)')
    args = parser.parse_args()

    if cv2 is None:
        print(f"[ERROR] opencv-python import failed: {_CV2_IMPORT_ERROR}")
        sys.exit(1)
    if tf is None:
        print(f"[ERROR] tensorflow import failed: {_TF_IMPORT_ERROR}")
        sys.exit(1)
    if rs is None:
        print(f"[ERROR] pyrealsense2 import failed: {_RS_IMPORT_ERROR}")
        print("        Install Intel RealSense Python bindings to run monitoring.")
        sys.exit(1)

    from models.movenet_pose_extractor import MoveNetPoseExtractor

    print("=" * 60)
    print("  Third Eye Shield -- Privacy-Preserving Wellness Monitor")
    print("  SSH Keyboard: R=Rec  S=Screenshot  E=Emotion  Q=Quit")
    print("=" * 60)

    root = Path(__file__).resolve().parents[1]

    # --- Load config.yaml --------------------------------------------------
    cfg = {}
    try:
        cp = root / 'config.yaml'
        if cp.exists():
            cfg = yaml.safe_load(cp.read_text()) or {}
    except Exception as e:
        print(f"[INIT] Config load warning: {e}")
        cfg = {}

    # --- Runtime config from env + YAML -----------------------------------
    # Priority: CLI args > environment variables > config.yaml
    wcfg = cfg.get('wellness', {}) if isinstance(cfg, dict) else {}

    # LLM config
    wcfg_llm = wcfg.get('llm', {}) if isinstance(wcfg, dict) else {}
    llm_enabled_cfg = bool(wcfg_llm.get('enabled', False))
    llm_endpoint_cfg = str(wcfg_llm.get('endpoint', '') or '').strip()
    llm_endpoint_env = os.getenv('THIRD_EYE_SHIELD_LLM_ENDPOINT', '').strip()
    llm_endpoint = (
        (args.llm_endpoint or '').strip()
        or llm_endpoint_env
        or (llm_endpoint_cfg if llm_enabled_cfg else '')
    )
    llm_checkin_sec_cfg = float(wcfg_llm.get('checkin_interval_sec', 7200))
    llm_msg_display_cfg = float(wcfg_llm.get('message_display_sec', 15))

    # --- Telegram notifier -------------------------------------------------
    telegram = None
    tg_token = (args.telegram_token or '').strip()
    tg_chat_ids = args.telegram_chat_ids or []
    # Actual Telegram init happens after config.yaml is loaded (see below)

    # --- Wellness trackers -------------------------------------------------
    posture_tracker = PostureTracker(alpha=0.15)
    sedentary_tracker = SedentaryTracker(threshold_sec=args.sedentary_minutes * 60)

    # --- Emotion module (opt-in, lazy loading) -----------------------------
    emotion_enabled = args.enable_emotion
    face_detector = None
    emotion_classifier = None
    emotion_tracker = None

    def _init_emotion():
        nonlocal face_detector, emotion_classifier, emotion_tracker
        from models.emotion_classifier import FaceDetector, EmotionClassifier, EmotionTracker
        print("[INIT] Emotion module (opt-in)...")
        face_detector = FaceDetector(min_detection_confidence=0.5)
        emotion_classifier = EmotionClassifier(
            model_path=args.emotion_model,
            feature_model_path=args.emotion_features_model,
            conf_threshold=0.35,
        )
        if args.emotion_profile:
            if emotion_classifier.load_profile(args.emotion_profile):
                print(f"[INIT] Few-shot profile loaded: {args.emotion_profile}")
            else:
                print(f"[WARN] Profile not found: {args.emotion_profile} -- using base model")
        elif args.emotion_calibration:
            emotion_classifier.load_calibration(args.emotion_calibration)
        emotion_tracker = EmotionTracker(window_sec=300)
        print("[INIT] Emotion module ready.")

    if emotion_enabled:
        try:
            _init_emotion()
        except Exception as e:
            print(f"[WARN] Emotion init failed: {e} -- running without emotion")
            emotion_enabled = False

    emotion_label = None
    emotion_conf = 0.0

    # --- MoveNet -----------------------------------------------------------
    print("[INIT] MoveNet Lightning int8 + DepthROI...")
    pose = MoveNetPoseExtractor(
        model_path='models/movenet_lightning_int8.tflite',
        conf_threshold=0.25
    )
    smoother = JointSmoother(alpha=0.55)

    # --- Autoencoder -------------------------------------------------------
    ae, ae_thr = None, 0.0912
    try:
        s1 = cfg.get('stage1', {})
        ae_thr = float(s1.get('reconstruction_threshold', ae_thr))
        mp = root / s1.get('model_path', 'models/depth_autoencoder_full.tflite')
        sz = tuple(s1.get('input_size', [64, 64]))
        if mp.exists():
            print(f"[INIT] AE: {mp.name}  threshold={ae_thr:.4f}")
            ae = DepthAutoencoder(mp, sz)
    except Exception as e:
        print(f"[INIT] AE disabled: {e}")

    # --- Action Classifier (RandomForest) ----------------------------------
    action_model = None
    action_classes = {}
    action_label = ""
    action_conf  = 0.0
    ACTION_CONF_THR = 0.38
    ACTION_VOTE_N   = 2
    THREAT_VOTE_N   = 3
    FALLING_CONF_THR = 0.55
    action_vote_buf = []
    ACTION_HOLD_SEC = 2.0
    action_last_t   = 0.0
    SEQ_LEN = 30
    skel_buf = collections.deque(maxlen=SEQ_LEN)
    prev_n_j = 0

    rf_path = root / 'models' / 'action_rf.pkl'
    if rf_path.exists():
        try:
            blob = pickle.load(open(rf_path, 'rb'))
            action_model = blob['model']
            action_classes = blob['classes']
            print(f"[INIT] Action RF: {blob['n_classes']} classes, {blob['n_samples']} training samples")
        except Exception as e:
            print(f"[INIT] Action RF disabled: {e}")

    # --- Telegram notifier (init after config/env are loaded) --------------
    try:
        from src.telegram_notifier import TelegramNotifier
        wcfg_tg = wcfg.get('telegram', {}) if isinstance(wcfg, dict) else {}
        tg_enabled = bool(wcfg_tg.get('enabled', False))

        tg_token_env = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
        tg_token = tg_token or tg_token_env or str(wcfg_tg.get('bot_token', '') or '').strip()

        env_chat_ids = []
        raw_ids = os.getenv('TELEGRAM_CHAT_IDS', '').strip()
        if raw_ids:
            env_chat_ids = [x.strip() for x in raw_ids.split(',') if x.strip()]
        elif os.getenv('TELEGRAM_CHAT_ID', '').strip():
            env_chat_ids = [os.getenv('TELEGRAM_CHAT_ID', '').strip()]

        if not tg_chat_ids:
            tg_chat_ids = env_chat_ids or wcfg_tg.get('chat_ids', [])
        tg_cooldown = wcfg_tg.get('cooldown_sec', 60)
        telegram = TelegramNotifier(
            bot_token=tg_token,
            chat_ids=tg_chat_ids,
            cooldown_sec=tg_cooldown,
            enabled=tg_enabled,
        )
    except Exception as e:
        print(f"[INIT] Telegram disabled: {e}")
        telegram = None

    # --- RealSense ---------------------------------------------------------
    print("[INIT] RealSense...")
    pipeline  = rs.pipeline()
    rscfg     = rs.config()
    rscfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    rscfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  30)
    pipeline.start(rscfg)
    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude,    2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

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
        print("[WARN] No interactive terminal - SSH keys disabled")

    def ssh_key():
        if not _has_tty:
            return None
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return sys.stdin.read(1)
        except Exception:
            pass
        return None

    # --- Dirs --------------------------------------------------------------
    clips_dir = Path('clips'); clips_dir.mkdir(exist_ok=True)
    screenshots_dir = Path('screenshots'); screenshots_dir.mkdir(exist_ok=True)

    # --- Wellness event log ------------------------------------------------
    log_dir = Path('logs/wellness')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"wellness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_f = open(log_path, 'w')
    log_f.write('timestamp,event,wellness_level,wellness_name,action,action_conf,'
                'posture_score,emotion,emotion_conf,sedentary_min,ae_mse\n')
    ALERT_COOLDOWN = 5.0
    last_alert_t   = 0.0

    cv2.namedWindow('Third Eye Shield', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Third Eye Shield', W, H)

    fps_avg = 25.0
    ae_mse  = None
    frame_n = 0
    t_prev  = time.time()
    wellness_level = WELLNESS_NORMAL
    wellness_reason = "starting"

    # Video writer state
    recording    = False
    video_writer = None
    rec_start    = 0.0
    rec_path     = None

    # LLM companion state
    llm_last_checkin = time.time()
    LLM_CHECKIN_SEC = llm_checkin_sec_cfg
    llm_message = ""
    llm_message_t = 0.0
    LLM_MSG_DISPLAY_SEC = llm_msg_display_cfg

    def _log_event(event, wl=None, wn=None):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        wl = wl if wl is not None else wellness_level
        wn = wn or WELLNESS_NAMES.get(wl, "")
        ps = f"{posture_tracker.score:.1f}" if posture_tracker.score is not None else ""
        em = emotion_label or ""
        ec = f"{emotion_conf:.2f}" if emotion_label else ""
        sm = f"{sedentary_tracker.sedentary_minutes:.1f}"
        mse_s = f"{ae_mse:.5f}" if ae_mse is not None else ""
        log_f.write(f"{ts},{event},{wl},{wn},{action_label},{action_conf:.3f},"
                    f"{ps},{em},{ec},{sm},{mse_s}\n")
        log_f.flush()

    def _send_llm_context(event_type="periodic"):
        """Send wellness context to LLM companion if endpoint configured."""
        nonlocal llm_message, llm_message_t, llm_last_checkin
        if not llm_endpoint:
            return
        try:
            import urllib.request, json
            context = {
                "event": event_type,
                "wellness_level": wellness_level,
                "wellness_name": WELLNESS_NAMES.get(wellness_level, ""),
                "action": action_label,
                "posture_score": posture_tracker.score,
                "emotion": emotion_label if emotion_enabled else None,
                "sedentary_minutes": round(sedentary_tracker.sedentary_minutes, 1),
                "emotion_enabled": emotion_enabled,
            }
            data = json.dumps({"context": context}).encode('utf-8')
            req = urllib.request.Request(
                llm_endpoint,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
                llm_message = result.get("response", "")
                llm_message_t = time.time()
                llm_last_checkin = time.time()
        except Exception as e:
            print(f"[LLM] Error: {e}")

    try:
      while _running:
        try:
            frames = pipeline.wait_for_frames(100)
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
        rgb = np.asanyarray(cf.get_data())

        # --- Depth: spatial filter + colourise ----------------------------
        df_f        = spatial.process(df)
        depth_color = np.asanyarray(colorizer.colorize(df_f).get_data())
        depth_bgr   = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
        disp        = cv2.resize(depth_bgr, (W, H))
        raw         = np.asanyarray(df.get_data())

        # --- Stage 1: AE (every 3 frames) --------------------------------
        if ae and frame_n % 3 == 0:
            try:
                ae_mse = ae.mse(raw)
            except Exception:
                ae_mse = None
        anom = ae_mse is not None and ae_mse < ae_thr

        # --- Stage 2+3: Skeleton + Action + Wellness (on anomaly) ---------
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
                draw_skeleton(disp, kps_disp)

            # --- Posture tracking (every frame with good skeleton) --------
            if n_j >= 5:
                posture_tracker.update(kps)

            # --- Emotion detection (opt-in, every 15 frames) --------------
            if emotion_enabled and face_detector and frame_n % 15 == 0 and n_j >= 3:
                try:
                    boxes = face_detector.detect(rgb)
                    if boxes:
                        x1, y1, x2, y2 = boxes[0]
                        face_crop = rgb[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            emotion_label, emotion_conf, _ = emotion_classifier.classify(face_crop)
                            emotion_tracker.update(emotion_label)
                except Exception:
                    pass  # emotion is best-effort, never crash the pipeline

            # Joint-stability gate
            joint_stable = (n_j >= 5 and not (prev_n_j >= 5 and n_j < prev_n_j * 0.5))
            prev_n_j = n_j

            # --- Action classification ------------------------------------
            if action_model and joint_stable:
                h_img, w_img = rgb.shape[:2]
                xy = np.stack([kps[:, 0] / w_img, kps[:, 1] / h_img], axis=-1)
                skel_buf.append(xy)
                if len(skel_buf) == SEQ_LEN and frame_n % 10 == 0:
                    try:
                        feats = extract_kinematic_features(list(skel_buf))
                        probs = action_model.predict_proba(feats.reshape(1, -1))[0]
                        pred_idx = np.argmax(probs)
                        pred_class = action_model.classes_[pred_idx]
                        pred_conf  = probs[pred_idx]
                        pred_name = action_classes.get(pred_class, f"class_{pred_class}")
                        eff_thr = FALLING_CONF_THR if pred_name == 'falling' else ACTION_CONF_THR
                        if pred_conf >= eff_thr:
                            action_vote_buf.append(pred_name)
                            need = THREAT_VOTE_N if pred_name in ALERT_ACTIONS else ACTION_VOTE_N
                            if len(action_vote_buf) > need:
                                action_vote_buf.pop(0)
                            if (len(action_vote_buf) >= need and
                                    len(set(action_vote_buf)) == 1):
                                action_label = pred_name
                                action_conf  = pred_conf
                                action_last_t = time.time()
                        else:
                            action_vote_buf.clear()
                            if action_label and action_label != "(idle)" and (time.time() - action_last_t) < ACTION_HOLD_SEC:
                                pass
                            else:
                                action_label = "(idle)"
                                action_conf = 0.0
                    except Exception:
                        pass

            # --- Sedentary tracking ---------------------------------------
            sedentary_tracker.update(action_label)

            # --- Compute wellness level -----------------------------------
            prolonged_neg = (emotion_enabled and emotion_tracker and
                            emotion_tracker.is_prolonged_negative)
            emo_for_level = emotion_label if (emotion_enabled and prolonged_neg) else None
            wellness_level, wellness_reason = compute_wellness_level(
                action_label, posture_tracker.score, sedentary_tracker,
                emotion_label=emo_for_level, emotion_enabled=emotion_enabled,
            )

            # --- Log wellness events --------------------------------------
            now = time.time()
            if wellness_level >= WELLNESS_CONCERN and now - last_alert_t > ALERT_COOLDOWN:
                last_alert_t = now
                event = "FALL_ALERT" if wellness_level == WELLNESS_ALERT else "CONCERN"
                _log_event(event)
                print(f"  [{event}] {wellness_reason}")
                # Notify LLM on alert/concern
                _send_llm_context(event_type=event.lower())
                # Notify family via Telegram
                if telegram and telegram.enabled:
                    tg_level = "alert" if wellness_level == WELLNESS_ALERT else "concern"
                    tg_ctx = {
                        "wellness_name": WELLNESS_NAMES.get(wellness_level, ""),
                        "action": action_label,
                        "posture_score": posture_tracker.score,
                        "sedentary_minutes": sedentary_tracker.sedentary_minutes,
                    }
                    telegram.send_alert(wellness_reason, level=tg_level, context=tg_ctx)

        else:
            # No person — clear state
            skel_buf.clear()
            action_vote_buf.clear()
            action_label = ""
            action_conf = 0.0
            emotion_label = None
            emotion_conf = 0.0
            smoother.reset()
            pose.reset_tracking()
            posture_tracker.reset()
            wellness_level = WELLNESS_NORMAL
            wellness_reason = "no person"

        # Log person enter/leave
        if anom and not getattr(main, '_was_anom', False):
            _log_event("PERSON_ENTERED")
            print(f"  [WELLNESS] Person entered (MSE={ae_mse:.4f})")
        if not anom and getattr(main, '_was_anom', False):
            _log_event("PERSON_LEFT")
            print(f"  [WELLNESS] Person left")
        main._was_anom = anom

        # --- Periodic LLM check-in ---------------------------------------
        if llm_endpoint and (time.time() - llm_last_checkin) > LLM_CHECKIN_SEC:
            _send_llm_context(event_type="periodic_checkin")

        # --- FPS ----------------------------------------------------------
        now     = time.time()
        fps_avg = 0.9*fps_avg + 0.1*(1.0/max(now-t_prev, 1e-6))
        t_prev  = now

        # --- HUD ----------------------------------------------------------
        wl_color = WELLNESS_COLORS_BGR.get(wellness_level, (255, 255, 255))
        wl_name = WELLNESS_NAMES.get(wellness_level, "")

        # Top-left: FPS + joints
        lock_tag = "  [LOCKED]" if pose.is_locked else ""
        cv2.putText(disp, f"FPS {fps_avg:.0f}  Joints {n_j}/17{lock_tag}",
                    (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(disp, f"FPS {fps_avg:.0f}  Joints {n_j}/17{lock_tag}",
                    (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Wellness level badge (top-right)
        badge = f" {wl_name.upper()} "
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bx = W - tw - 16
        cv2.rectangle(disp, (bx-4, 6), (W-8, 6+th+12), wl_color, -1)
        cv2.putText(disp, badge, (bx, 6+th+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Action label
        if anom:
            disp_label = action_label if action_label else "(idle)"
            is_alert = disp_label in ALERT_ACTIONS
            if is_alert:
                action_col = (0, 0, 255)
            elif disp_label == "(idle)":
                action_col = (160, 160, 160)
            else:
                action_col = (0, 255, 255)
            conf_str = f" ({action_conf:.0%})" if action_conf > 0 else ""
            atxt = f"Action: {disp_label}{conf_str}"
            cv2.putText(disp, atxt, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(disp, atxt, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_col, 2)

            # Fall alert overlay
            if is_alert and action_conf >= 0.45:
                cv2.putText(disp, "!! FALL ALERT !!", (W//2-120, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5)
                cv2.putText(disp, "!! FALL ALERT !!", (W//2-120, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # Posture score (bottom-left area)
        ps = posture_tracker.score
        if ps is not None and anom:
            ps_col = (0, 200, 0) if ps >= 65 else (0, 180, 255) if ps >= 35 else (0, 0, 255)
            cv2.putText(disp, f"Posture: {ps:.0f}/100", (8, H-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(disp, f"Posture: {ps:.0f}/100", (8, H-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ps_col, 2)

        # Emotion (only if enabled)
        if emotion_enabled and emotion_label and anom:
            em_txt = f"Emotion: {emotion_label} ({emotion_conf:.0%})"
            cv2.putText(disp, em_txt, (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3)
            cv2.putText(disp, em_txt, (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,200,0), 2)
        elif not emotion_enabled and anom:
            cv2.putText(disp, "Emotion: [off]", (8, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,120), 1)

        # Sedentary timer
        sed_min = sedentary_tracker.sedentary_minutes
        if sed_min >= 5 and anom:
            sed_col = (0, 180, 255) if sedentary_tracker.is_sedentary else (180, 180, 180)
            cv2.putText(disp, f"Inactive: {sed_min:.0f} min", (8, H-65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, sed_col, 2)

        # AE status
        if ae_mse is not None:
            col   = (0,0,255) if anom else (0,200,80)
            label = "Person" if anom else "Empty"
            cv2.putText(disp, f"AE: {label}  {ae_mse:.4f}",
                        (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
            cv2.putText(disp, f"AE: {label}  {ae_mse:.4f}",
                        (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        # LLM message overlay
        if llm_message and (time.time() - llm_message_t) < LLM_MSG_DISPLAY_SEC:
            # Chat bubble at bottom-right
            lines = [llm_message[i:i+40] for i in range(0, len(llm_message), 40)]
            for li, line in enumerate(lines[-3:]):  # show last 3 lines
                y = H - 80 - (2 - li) * 22
                cv2.putText(disp, line, (W//2, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Wellness-coloured border
        cv2.rectangle(disp, (0,0), (W-1,H-1), wl_color, 6)

        # Third Eye Shield branding
        brand_txt = "Third Eye Shield"
        (bw, _), _ = cv2.getTextSize(brand_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(disp, brand_txt, (W - bw - 10, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        # Recording indicator
        if recording:
            elapsed = time.time() - rec_start
            mins, secs = int(elapsed // 60), int(elapsed % 60)
            if int(elapsed * 2) % 2 == 0:
                cv2.circle(disp, (W - 30, 25), 10, (0, 0, 255), -1)
            cv2.putText(disp, f"REC {mins:02d}:{secs:02d}", (W - 150, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            if video_writer is not None:
                video_writer.write(disp)

        cv2.imshow('Third Eye Shield', disp)
        wk = cv2.waitKey(1) & 0xFF
        if wk in (ord('q'), 27):
            break

        # --- SSH keyboard --------------------------------------------------
        k = ssh_key()
        if k is not None:
            if k in ('r', 'R'):
                if not recording:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    rec_path = clips_dir / f"third_eye_shield_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(rec_path), fourcc, 15.0, (W, H))
                    rec_start = time.time()
                    recording = True
                    print(f"  [REC] Recording: {rec_path.name}")
                else:
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    print(f"  [REC] Stopped -> {rec_path.name}")
            elif k in ('s', 'S'):
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                sp = screenshots_dir / f"third_eye_shield_{ts}.png"
                cv2.imwrite(str(sp), disp)
                print(f"  [SCREENSHOT] {sp.name}")
            elif k in ('e', 'E'):
                # Toggle emotion detection
                emotion_enabled = not emotion_enabled
                if emotion_enabled and face_detector is None:
                    try:
                        _init_emotion()
                    except Exception as e:
                        print(f"  [EMOTION] Init failed: {e}")
                        emotion_enabled = False
                if emotion_enabled:
                    print("  [EMOTION] ON -- face processing active (opt-in)")
                else:
                    emotion_label = None
                    emotion_conf = 0.0
                    print("  [EMOTION] OFF -- no face processing (privacy mode)")
            elif k in ('l', 'L'):
                if pose.is_locked:
                    pose.unlock()
                    print("  [LOCK] Unlocked")
                else:
                    pose.lock_on(depth_frame=raw)
                    print("  [LOCK] Locked onto target")
            elif k in ('q', 'Q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
        if video_writer:
            video_writer.release()
        if face_detector:
            try:
                face_detector.close()
            except Exception:
                pass
        log_f.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[Third Eye Shield] Stopped.  Log: {log_path}")


if __name__ == '__main__':
    main()
