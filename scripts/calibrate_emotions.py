#!/usr/bin/env python3
"""
Third Eye Shield -- Per-User Emotion Registration (Few-Shot)
=====================================================

Registers a user's facial expressions for personalized emotion detection
using prototypical few-shot learning.

For each of 7 emotions, the user provides face samples. These are converted
to 128-dim embeddings and stored as the user's emotion profile. At inference
time, new faces are classified by cosine similarity to these prototypes.

Usage:
    python3 scripts/calibrate_emotions.py --user-id UNCLE_TAN
    python3 scripts/calibrate_emotions.py --user-id UNCLE_TAN --samples 15
    python3 scripts/calibrate_emotions.py --user-id UNCLE_TAN --add happy
                                          # add more samples for one emotion

Requires: Run setup_emotion_model.py first to create TFLite models.
"""
import os, sys, signal, time, argparse, select, termios, tty
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.emotion_classifier import (
    FaceDetector, EmotionClassifier, EMOTION_LABELS, FEATURE_DIM,
)

EMOTIONS = EMOTION_LABELS
SAMPLES_PER_EMOTION = 10
W, H = 800, 480

_running = True
def _stop(sig, frame):
    global _running; _running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

PROMPTS = {
    'angry':    "Show an ANGRY face (furrowed brow, clenched jaw)",
    'disgust':  "Show DISGUST (wrinkle your nose, raise upper lip)",
    'fear':     "Show FEAR (wide eyes, open mouth slightly)",
    'happy':    "Show a HAPPY face (big smile, eyes crinkle)",
    'sad':      "Show a SAD face (frown, droopy eyes)",
    'surprise': "Show SURPRISE (raised eyebrows, open mouth wide)",
    'neutral':  "Show a NEUTRAL face (relaxed, no expression)",
}


def main():
    parser = argparse.ArgumentParser(
        description="Third Eye Shield Few-Shot Emotion Registration")
    parser.add_argument('--user-id', required=True,
                        help="User identifier (e.g. UNCLE_TAN)")
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_EMOTION,
                        help=f'Samples per emotion (default: {SAMPLES_PER_EMOTION})')
    parser.add_argument('--add', type=str, default=None,
                        help='Add samples for a single emotion (e.g. --add happy)')
    parser.add_argument('--model', type=str, default=None,
                        help='Classifier TFLite path (auto-detected)')
    parser.add_argument('--feature-model', type=str, default=None,
                        help='Feature extractor TFLite path (auto-detected)')
    args = parser.parse_args()

    user_id = args.user_id.strip().replace(' ', '_')

    # Validate --add argument
    if args.add:
        if args.add not in EMOTION_LABELS:
            print(f"[ERROR] Unknown emotion: {args.add}")
            print(f"        Valid: {', '.join(EMOTION_LABELS)}")
            sys.exit(1)
        emotions_to_register = [args.add]
    else:
        emotions_to_register = list(EMOTIONS)

    print("=" * 60)
    print(f"  Third Eye Shield -- Emotion Registration for: {user_id}")
    if args.add:
        print(f"  Adding {args.samples} samples for: {args.add}")
    else:
        print(f"  {len(emotions_to_register)} emotions x {args.samples} samples")
    print("=" * 60)

    # --- Init models ---------------------------------------------------
    print("[INIT] Face detector (MediaPipe)...")
    face_det = FaceDetector(min_detection_confidence=0.5)

    print("[INIT] Emotion classifier + feature extractor...")
    emotion_clf = EmotionClassifier(
        model_path=args.model,
        feature_model_path=args.feature_model,
        conf_threshold=0.0,  # no threshold during registration
    )

    if emotion_clf._feat_extractor is None:
        print("[ERROR] Feature extractor not available.")
        print("        Run: python3 scripts/setup_emotion_model.py")
        sys.exit(1)

    # Load existing profile if adding to it
    if args.add:
        loaded = emotion_clf.load_profile(user_id)
        if loaded:
            stats = emotion_clf.get_profile_stats()
            print(f"[INFO] Existing profile: {stats}")
        else:
            print(f"[INFO] No existing profile -- creating new for {user_id}")

    # --- Camera --------------------------------------------------------
    print("[INIT] RealSense...")
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    pipeline.start(cfg)
    for _ in range(15):
        pipeline.wait_for_frames()

    # Terminal setup
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    cv2.namedWindow("Register", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Register", W, H)

    total_registered = 0

    try:
        for emo_idx, emotion in enumerate(emotions_to_register):
            if not _running:
                break

            emotion_idx = EMOTION_LABELS.index(emotion)
            prompt = PROMPTS.get(emotion, f"Show: {emotion}")
            captured = 0
            state = "IDLE"
            countdown_start = 0
            COUNTDOWN_SEC = 3

            progress = (f"Emotion {emo_idx+1}/{len(emotions_to_register)}"
                        if not args.add else f"Adding: {emotion}")

            while _running and captured < args.samples:
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
                disp = cv2.resize(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (W, H))

                # Face detection
                boxes = face_det.detect(rgb)
                has_face = len(boxes) > 0

                if has_face:
                    x1, y1, x2, y2 = boxes[0]
                    sx, sy = W / rgb.shape[1], H / rgb.shape[0]
                    dx1, dy1 = int(x1*sx), int(y1*sy)
                    dx2, dy2 = int(x2*sx), int(y2*sy)
                    cv2.rectangle(disp, (dx1, dy1), (dx2, dy2),
                                  (0, 255, 0), 2)

                    # Show base model prediction for reference
                    face_crop = rgb[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        label, conf, _ = emotion_clf._classify_base(face_crop)
                        cv2.putText(disp, f"Base: {label} ({conf:.0%})",
                                    (dx1, dy1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (200, 200, 0), 1)

                # Header
                cv2.putText(disp, f"{progress}: {emotion.upper()}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)
                cv2.putText(disp, prompt, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)
                cv2.putText(disp, f"Captured: {captured}/{args.samples}",
                            (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                if state == "IDLE":
                    if has_face:
                        cv2.putText(disp, "Press SPACE when ready",
                                    (10, H-45), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (0, 200, 200), 2)
                    else:
                        cv2.putText(disp, "No face -- look at camera",
                                    (10, H-45), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (0, 0, 255), 2)

                elif state == "COUNTDOWN":
                    remaining = COUNTDOWN_SEC - (time.time() - countdown_start)
                    if remaining > 0:
                        cv2.putText(disp, f"{int(remaining)+1}",
                                    (W//2-30, H//2+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4.0,
                                    (0, 255, 255), 8)
                    else:
                        state = "CAPTURING"

                elif state == "CAPTURING":
                    cv2.rectangle(disp, (0, 0), (W-1, H-1),
                                  (0, 0, 255), 8)
                    if has_face:
                        x1, y1, x2, y2 = boxes[0]
                        face_crop = rgb[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            ok = emotion_clf.register_sample(
                                user_id, emotion_idx, face_crop)
                            if ok:
                                captured += 1
                                total_registered += 1
                                cv2.putText(
                                    disp, f"Registered #{captured}!",
                                    (W//2-100, H//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 0), 3)
                                time.sleep(0.3)
                                if captured < args.samples:
                                    state = "CAPTURING"
                                else:
                                    state = "IDLE"
                            else:
                                cv2.putText(
                                    disp, "Feature extraction failed",
                                    (W//2-120, H//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                                state = "IDLE"
                    else:
                        cv2.putText(disp, "Face lost! Look at camera.",
                                    (W//2-120, H//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        state = "IDLE"

                cv2.imshow("Register", disp)
                wk = cv2.waitKey(1) & 0xFF
                if wk == 27 or wk == ord('q'):
                    _running = False
                    break
                if wk == ord(' ') and state == "IDLE" and has_face:
                    state = "COUNTDOWN"
                    countdown_start = time.time()

                # SSH key check
                if select.select([sys.stdin], [], [], 0) == (
                        [sys.stdin], [], []):
                    ch = sys.stdin.read(1)
                    if ch == ' ' and state == "IDLE" and has_face:
                        state = "COUNTDOWN"
                        countdown_start = time.time()
                    elif ch in ('q', 'Q'):
                        _running = False
                        break

            print(f"  [{emotion}] Registered {captured} samples")

        # --- Save profile ----------------------------------------------
        if total_registered > 0:
            emotion_clf.save_profile(user_id)
            stats = emotion_clf.get_profile_stats()
            print(f"\n[DONE] Profile for {user_id}:")
            for emo, count in stats.items():
                bar = "#" * count
                print(f"  {emo:>10s}: {count:3d} {bar}")
            print(f"\n  To use:")
            print(f"    python3 scripts/wellness_monitor.py "
                  f"--enable-emotion --emotion-profile {user_id}")
            print(f"\n  To add more samples later:")
            print(f"    python3 scripts/calibrate_emotions.py "
                  f"--user-id {user_id} --add happy")
        else:
            print("\n[WARN] No samples registered.")

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        face_det.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Third Eye Shield] Registration session ended.")


if __name__ == '__main__':
    main()
