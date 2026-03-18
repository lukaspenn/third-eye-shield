#!/usr/bin/env python3
"""
Touchscreen Launcher - 800x480 Pi Display  (v2 - premium blue UI)
Two-mode app: Live Detection + Clips Gallery, controlled by touch.

Usage:
    DISPLAY=:0 python3 scripts/touchscreen_launcher.py

Touch controls:
    HOME SCREEN:  Tap "LIVE DETECTION" or "CLIPS GALLERY"
    LIVE MODE:    Tap BACK (bottom-left), REC (bottom-right), or screen (screenshot)
    GALLERY:      Tap clip thumbnail to play, X icon to delete, arrows to scroll
    PLAYBACK:     Speed buttons (0.5x 1x 2x), BACK to stop
"""
import os, sys, signal, time, pickle, collections, glob
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

sys.stdout.reconfigure(line_buffering=True)

_running = True
def _stop(sig, frame):
    global _running; _running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

W, H = 800, 480
JOINT_CONF_THR = 0.25
WIN = 'DepthGuard'

MOVENET_BONES = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]
THREAT_ACTIONS = {'punching/slapping', 'pushing other person', 'kicking something', 'falling'}

# ── Blue premium palette (BGR) ────────────────────────────────────────────────
BG_DARK       = (20, 18, 15)
BG_CARD       = (50, 40, 30)
BLUE_PRIMARY  = (230, 160, 30)
BLUE_GLOW     = (255, 200, 80)
BLUE_DARK     = (180, 120, 15)
BLUE_DIM      = (120, 80, 10)
WHITE         = (255, 255, 255)
GRAY          = (160, 160, 160)
GRAY_DIM      = (100, 100, 100)
RED           = (60, 60, 230)
GREEN         = (80, 200, 80)
CYAN          = (220, 220, 60)

# ── Drawing primitives ────────────────────────────────────────────────────────

def _gradient_v(img, y1, y2, col_top, col_bot):
    """Fill horizontal band y1..y2 with a vertical gradient."""
    for y in range(max(0, y1), min(img.shape[0], y2)):
        t = (y - y1) / max(1, y2 - y1)
        c = tuple(int(col_top[i]*(1-t) + col_bot[i]*t) for i in range(3))
        img[y, :] = c

def _fill_rounded(img, pt1, pt2, color, r):
    x1, y1 = pt1; x2, y2 = pt2
    r = min(r, (x2-x1)//2, (y2-y1)//2)
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
    cv2.circle(img, (x1+r, y1+r), r, color, -1)
    cv2.circle(img, (x2-r, y1+r), r, color, -1)
    cv2.circle(img, (x1+r, y2-r), r, color, -1)
    cv2.circle(img, (x2-r, y2-r), r, color, -1)

def _rounded_rect_shadow(img, pt1, pt2, fill, radius=16, shadow_off=4, shadow_blur=7):
    """Draw a rounded rectangle with a drop shadow underneath."""
    x1, y1 = pt1; x2, y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    sh = np.zeros_like(img)
    _fill_rounded(sh, (x1+shadow_off, y1+shadow_off), (x2+shadow_off, y2+shadow_off), (40,40,40), r)
    blurred = cv2.GaussianBlur(sh, (shadow_blur*2+1, shadow_blur*2+1), shadow_blur)
    mask = blurred.astype(np.float32) / 255.0
    img[:] = (img.astype(np.float32) * (1 - mask*0.5) + blurred.astype(np.float32) * 0.3).clip(0,255).astype(np.uint8)
    _fill_rounded(img, pt1, pt2, fill, r)

def _rounded_rect_border(img, pt1, pt2, color, radius=16, thickness=2):
    x1, y1 = pt1; x2, y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

def _pill_btn(img, cx, cy, text, w=90, h=34, fill=(60,50,40), text_col=WHITE, border=None):
    """Draw a pill-shaped button centred at (cx,cy). Returns (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = cx-w//2, cy-h//2, cx+w//2, cy+h//2
    _fill_rounded(img, (x1, y1), (x2, y2), fill, h//2)
    if border:
        _rounded_rect_border(img, (x1, y1), (x2, y2), border, h//2, 2)
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
    cv2.putText(img, text, (cx - tw//2, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 2, cv2.LINE_AA)
    return (x1, y1, x2, y2)

def in_rect(x, y, x1, y1, x2, y2):
    return x1 <= x <= x2 and y1 <= y <= y2


# ── Model helpers ─────────────────────────────────────────────────────────────
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
            cv2.line(img, px(a), px(b), BLUE_GLOW, 3)
    for j in range(17):
        if kps[j,2]>JOINT_CONF_THR:
            p = px(j); r = 8 if j==0 else 5
            cv2.circle(img, p, r+2, (0,0,0), -1)
            cv2.circle(img, p, r, BLUE_PRIMARY, -1)

class AE:
    def __init__(self, path, size=(64,64)):
        self.size = size
        it = tf.lite.Interpreter(model_path=str(path)); it.allocate_tensors()
        self.it = it
        self.inp = it.get_input_details()[0]['index']
        self.out = it.get_output_details()[0]['index']
    def mse(self, raw):
        d = cv2.resize(raw, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        d = np.clip((d-100)/9900.0, 0, 1)[None,...,None]
        self.it.set_tensor(self.inp, d); self.it.invoke()
        rec = self.it.get_tensor(self.out)[0,...,0]
        return float(np.mean((d[0,...,0]-rec)**2))


def get_clip_list(clips_dir):
    files = sorted(Path(clips_dir).glob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for f in files:
        try:
            if f.stat().st_size < 100_000:  # skip tiny/corrupt clips (<100KB)
                continue
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                continue
            fps_v = cap.get(cv2.CAP_PROP_FPS) or 15
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if frames < 5:  # skip clips with fewer than 5 frames
                cap.release()
                continue
            dur = frames / max(fps_v, 1)
            ret, thumb = cap.read()
            cap.release()
            if ret:
                thumb = cv2.resize(thumb, (220, 132))
            else:
                thumb = np.zeros((132, 220, 3), dtype=np.uint8)
            result.append((f, dur, thumb))
        except Exception:
            continue
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  HOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════
def draw_home(n_clips=0):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    _gradient_v(img, 0, H, (20, 18, 15), (35, 30, 25))

    # Top accent line
    cv2.line(img, (0, 0), (W, 0), BLUE_PRIMARY, 3)

    # Title - centered
    title = "DEPTH-BASED ANOMALY DETECTION"
    tw_t = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
    cv2.putText(img, title, (W//2 - tw_t//2, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLUE_GLOW, 2, cv2.LINE_AA)
    subtitle = "Privacy-preserving Security Surveillance"
    tw_s = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    cv2.putText(img, subtitle, (W//2 - tw_s//2, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY_DIM, 1, cv2.LINE_AA)

    # ── Cards (centered) ──────────────────────────────────────────────────
    card_w, card_h = 330, 225
    gap = 30
    total_cw = 2 * card_w + gap
    x0 = (W - total_cw) // 2
    y0 = 120
    lx1, ly1, lx2, ly2 = x0, y0, x0 + card_w, y0 + card_h
    rx1, ry1, rx2, ry2 = lx2 + gap, y0, lx2 + gap + card_w, y0 + card_h
    lcx, rcx = (lx1 + lx2) // 2, (rx1 + rx2) // 2

    # ── LIVE DETECTION card ───────────────────────────────────────────────
    _rounded_rect_shadow(img, (lx1, ly1), (lx2, ly2), BG_CARD, 18)
    _rounded_rect_border(img, (lx1, ly1), (lx2, ly2), BLUE_DIM, 18, 2)
    # Play icon with glow
    icy = ly1 + 90
    pts = np.array([[lcx-35, icy-40], [lcx+40, icy], [lcx-35, icy+40]], np.int32)
    glow = img.copy()
    cv2.fillPoly(glow, [pts], BLUE_GLOW)
    cv2.addWeighted(glow, 0.15, img, 0.85, 0, img)
    cv2.fillPoly(img, [pts], BLUE_PRIMARY)
    for lbl, font_sc, yoff, col, th in [("LIVE", 0.95, ly2-50, WHITE, 2), ("DETECTION", 0.65, ly2-15, GRAY, 1)]:
        tw_l = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_sc, th)[0][0]
        cv2.putText(img, lbl, (lcx - tw_l//2, yoff),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sc, col, th, cv2.LINE_AA)

    # ── CLIPS GALLERY card ────────────────────────────────────────────────
    _rounded_rect_shadow(img, (rx1, ry1), (rx2, ry2), BG_CARD, 18)
    _rounded_rect_border(img, (rx1, ry1), (rx2, ry2), BLUE_DIM, 18, 2)
    gx = rcx - 32; gy = ry1 + 70
    for r in range(2):
        for c in range(2):
            bx = gx + c*35; by = gy + r*35
            _fill_rounded(img, (bx, by), (bx+28, by+28), BLUE_PRIMARY, 5)
    for lbl, font_sc, yoff, col, th in [("CLIPS", 0.95, ry2-50, WHITE, 2)]:
        tw_l = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_sc, th)[0][0]
        cv2.putText(img, lbl, (rcx - tw_l//2, yoff),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sc, col, th, cv2.LINE_AA)
    clip_txt = f"GALLERY ({n_clips})" if n_clips else "GALLERY"
    tw_c = cv2.getTextSize(clip_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
    cv2.putText(img, clip_txt, (rcx - tw_c//2, ry2-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, GRAY, 1, cv2.LINE_AA)

    # ── Bottom bar ────────────────────────────────────────────────────────
    cv2.line(img, (55, 390), (745, 390), BLUE_DIM, 1, cv2.LINE_AA)
    stats = "Edge AI  |  >= 90% accuracy  |  Raspberry Pi  |  By Kien, Ize, Enzo"
    tw_st = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    cv2.putText(img, stats, (W//2 - tw_st//2, 425),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY_DIM, 1, cv2.LINE_AA)
    prompt = "Tap a card to begin"
    tw_p = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
    cv2.putText(img, prompt, (W//2 - tw_p//2, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, BLUE_PRIMARY, 1, cv2.LINE_AA)

    return img


# ══════════════════════════════════════════════════════════════════════════════
#  GALLERY SCREEN
# ══════════════════════════════════════════════════════════════════════════════
def draw_gallery(clips, page=0, delete_mode_idx=-1):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    _gradient_v(img, 0, H, (20, 18, 15), (35, 30, 25))
    cv2.line(img, (0, 0), (W, 0), BLUE_PRIMARY, 3)

    # Header
    _pill_btn(img, 60, 30, "< BACK", w=100, h=36, fill=BG_CARD, border=BLUE_DIM)
    cv2.putText(img, "CLIP GALLERY", (310, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE_GLOW, 2, cv2.LINE_AA)

    per_page = 6
    start = page * per_page
    page_clips = clips[start:start+per_page]
    total_pages = max(1, (len(clips) + per_page - 1) // per_page)

    thumb_w, thumb_h = 220, 132
    gap_x, gap_y = 30, 20
    x_off, y_off = 45, 62

    rects = []
    del_rects = []

    for i, (path, dur, thumb) in enumerate(page_clips):
        col = i % 3
        row = i // 3
        x1 = x_off + col * (thumb_w + gap_x)
        y1 = y_off + row * (thumb_h + gap_y + 34)
        x2 = x1 + thumb_w
        y2 = y1 + thumb_h

        # Card background
        _fill_rounded(img, (x1-4, y1-4), (x2+4, y2+24), BG_CARD, 8)
        _rounded_rect_border(img, (x1-4, y1-4), (x2+4, y2+24), BLUE_DIM, 8, 1)

        # Thumbnail
        img[y1:y2, x1:x2] = thumb

        # Duration badge
        mins = int(dur // 60); secs = int(dur % 60)
        dur_txt = f"{mins:02d}:{secs:02d}"
        dtw = cv2.getTextSize(dur_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.rectangle(img, (x1, y2-22), (x1+dtw+10, y2), (0,0,0), -1)
        cv2.putText(img, dur_txt, (x1+5, y2-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE_GLOW, 1, cv2.LINE_AA)

        # Name
        name = path.stem[:25]
        cv2.putText(img, name, (x1+2, y2+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1, cv2.LINE_AA)

        # Delete icon (X button top-right of thumbnail)
        dx1, dy1, dx2, dy2 = x2-24, y1, x2, y1+24
        _fill_rounded(img, (dx1, dy1), (dx2, dy2), (40, 30, 130), 4)
        cv2.putText(img, "X", (dx1+5, dy2-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
        del_rects.append((dx1, dy1, dx2, dy2, start + i))

        # Confirm delete overlay
        if delete_mode_idx == start + i:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 180), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.putText(img, "DELETE?", (x1+60, y1+55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
            cv2.putText(img, "Tap X again", (x1+58, y1+82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,255), 1, cv2.LINE_AA)

        rects.append((x1, y1, x2, y2+20, start + i))

    # Pagination
    if total_pages > 1:
        pg_txt = f"{page+1} / {total_pages}"
        tw = cv2.getTextSize(pg_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(img, pg_txt, (W//2 - tw//2, H-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY_DIM, 1, cv2.LINE_AA)
        if page > 0:
            _pill_btn(img, 120, H-32, "< PREV", w=100, h=32, fill=BG_CARD, border=BLUE_DIM)
        if page < total_pages - 1:
            _pill_btn(img, W-120, H-32, "NEXT >", w=100, h=32, fill=BG_CARD, border=BLUE_DIM)

    if not clips:
        cv2.putText(img, "No clips recorded yet.", (248, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1, cv2.LINE_AA)
        cv2.putText(img, "Use Live Detection to record clips.", (175, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE_PRIMARY, 1, cv2.LINE_AA)

    return img, rects, total_pages, del_rects


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYBACK HUD
# ══════════════════════════════════════════════════════════════════════════════
def draw_play_hud(frame, pos_sec, total_sec, speed_idx, speeds, clip_name):
    h, w = frame.shape[:2]
    btns = {}

    # Bottom-only bar - keeps top clear so baked-in metrics stay visible
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-70), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Row 1: BACK | clip name | speed buttons   (y center ~ h-52)
    row1_y = h - 52
    btns['back'] = _pill_btn(frame, 55, row1_y, "< BACK", w=90, h=28, fill=BG_CARD, border=BLUE_DIM)

    nw = cv2.getTextSize(clip_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(frame, clip_name, (w//2 - nw//2, row1_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE_GLOW, 1, cv2.LINE_AA)

    speed_labels = ["0.5x", "1x", "2x"]
    sx = w - 170
    for si, sl in enumerate(speed_labels):
        active = (si == speed_idx)
        f = BLUE_PRIMARY if active else BG_CARD
        b = BLUE_GLOW if active else BLUE_DIM
        btns[f'speed_{si}'] = _pill_btn(frame, sx + si*60, row1_y, sl, w=50, h=28, fill=f, border=b,
                                          text_col=WHITE if active else GRAY)

    # Row 2: progress bar   (y center ~ h-18)
    bar_x1, bar_x2, bar_y = 120, w-120, h-18
    cv2.line(frame, (bar_x1, bar_y), (bar_x2, bar_y), GRAY_DIM, 4, cv2.LINE_AA)
    if total_sec > 0:
        prog_x = int(bar_x1 + (bar_x2 - bar_x1) * min(pos_sec / total_sec, 1.0))
        cv2.line(frame, (bar_x1, bar_y), (prog_x, bar_y), BLUE_PRIMARY, 4, cv2.LINE_AA)
        cv2.circle(frame, (prog_x, bar_y), 7, BLUE_GLOW, -1, cv2.LINE_AA)

    def fmt(s): return f"{int(s//60):02d}:{int(s%60):02d}"
    cv2.putText(frame, fmt(pos_sec), (bar_x1-95, bar_y+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, fmt(total_sec), (bar_x2+10, bar_y+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    return btns


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    from models.movenet_pose_extractor import MoveNetPoseExtractor

    root = Path(__file__).resolve().parents[1]
    clips_dir = root / 'clips'
    clips_dir.mkdir(exist_ok=True)
    screenshots_dir = root / 'screenshots'
    screenshots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  TOUCHSCREEN LAUNCHER  v2  |  800x480")
    print("=" * 60)

    print("[INIT] MoveNet Lightning int8 + DepthROI...")
    pose = MoveNetPoseExtractor(model_path='models/movenet_lightning_int8.tflite', conf_threshold=0.25)
    smoother = JointSmoother(alpha=0.55)

    cfg = yaml.safe_load(open(root/'config.yaml')) or {}
    s1 = cfg.get('stage1', {})
    ae_thr = float(s1.get('reconstruction_threshold', 0.0912))
    ae = AE(root/s1.get('model_path','models/depth_autoencoder_full.tflite'),
            tuple(s1.get('input_size',[64,64])))
    print(f"[INIT] AE: threshold={ae_thr:.4f}")

    action_model, action_classes = None, {}
    rf_path = root/'models'/'action_rf.pkl'
    if rf_path.exists():
        blob = pickle.load(open(rf_path,'rb'))
        action_model, action_classes = blob['model'], blob['classes']
        print(f"[INIT] Action RF: {blob['n_classes']} classes")

    print("[INIT] RealSense...")
    rs_pipeline = rs.pipeline()
    rscfg = rs.config()
    rscfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    rscfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    rs_pipeline.start(rscfg)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    for _ in range(15):
        rs_pipeline.wait_for_frames()
    print("[INIT] Ready.\n")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Touch
    touch = {'x': -1, 'y': -1, 'clicked': False}
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            touch['x'], touch['y'], touch['clicked'] = x, y, True
    cv2.setMouseCallback(WIN, on_mouse)

    def pop_click():
        if touch['clicked']:
            touch['clicked'] = False
            return touch['x'], touch['y']
        return None

    # State
    MODE_HOME, MODE_LIVE, MODE_GALLERY, MODE_PLAY = 0, 1, 2, 3
    mode = MODE_HOME

    frame_n = 0; ae_mse = None; anom = False; n_j = 0
    fps_avg = 20.0; t_prev = time.time()
    action_label, action_conf = "", 0.0
    ACTION_CONF_THR, ACTION_VOTE_N = 0.38, 2
    THREAT_VOTE_N = 3              # extra votes required for threat actions
    FALLING_CONF_THR = 0.55        # higher bar for 'falling' to reduce false positives
    action_vote_buf = []
    ACTION_HOLD_SEC = 2.0; action_last_t = 0.0
    SEQ_LEN = 30
    skel_buf = collections.deque(maxlen=SEQ_LEN)
    prev_n_j = 0                   # joint count from previous frame (dropout detection)

    recording = False; video_writer = None; rec_start = 0.0; rec_path = None

    gallery_page = 0; gallery_clips = []; gallery_rects = []
    gallery_del_rects = []; gallery_total_pages = 1
    delete_confirm_idx = -1

    play_cap = None; play_clip_name = ""; play_total_sec = 0.0; play_fps = 15.0
    speeds = [0.5, 1.0, 2.0]; speed_idx = 1
    play_btns = {}; play_fail_count = 0

    try:
        while _running:
            # ── HOME ──────────────────────────────────────────────────────
            if mode == MODE_HOME:
                n_clips = len(list(clips_dir.glob('*.mp4')))
                cv2.imshow(WIN, draw_home(n_clips))
                cv2.waitKey(30)
                click = pop_click()
                if click:
                    cx, cy = click
                    if in_rect(cx, cy, 55, 120, 385, 345):
                        mode = MODE_LIVE; frame_n = 0; ae_mse = None
                        skel_buf.clear(); action_vote_buf.clear()
                        action_label, action_conf = "", 0.0
                        print("[MODE] -> Live Detection")
                    elif in_rect(cx, cy, 415, 120, 745, 345):
                        mode = MODE_GALLERY; gallery_page = 0; delete_confirm_idx = -1
                        gallery_clips = get_clip_list(clips_dir)
                        print(f"[MODE] -> Gallery ({len(gallery_clips)} clips)")
                continue

            # ── LIVE DETECTION ────────────────────────────────────────────
            if mode == MODE_LIVE:
                try:
                    frames = rs_pipeline.wait_for_frames(100)
                except Exception:
                    continue
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

                if frame_n % 3 == 0:
                    ae_mse = ae.mse(raw)
                anom = ae_mse is not None and ae_mse < ae_thr

                n_j = 0
                if anom:
                    kps_raw, _, _ = pose.extract(rgb, draw=False, depth_frame=raw)
                    kps = smoother(kps_raw)
                    sx, sy = W/rgb.shape[1], H/rgb.shape[0]
                    kps_d = kps.copy(); kps_d[:,0]*=sx; kps_d[:,1]*=sy
                    n_j = int(np.sum(kps[:,2]>JOINT_CONF_THR))
                    if n_j >= 1:
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
                                # Higher confidence bar for 'falling'
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
                                        pass
                                    else:
                                        action_label, action_conf = "(idle)", 0.0
                            except Exception:
                                pass
                else:
                    skel_buf.clear(); action_vote_buf.clear()
                    action_label, action_conf = "", 0.0
                    smoother.reset()
                    pose.reset_tracking()

                now = time.time()
                fps_avg = 0.9*fps_avg + 0.1*(1.0/max(now-t_prev,1e-6))
                t_prev = now

                # HUD
                border_col = RED if anom else GREEN
                cv2.rectangle(disp, (0,0), (W-1,H-1), border_col, 5)

                lock_tag = "  [LOCKED]" if pose.is_locked else ""
                cv2.putText(disp, f"FPS {fps_avg:.0f}  Joints {n_j}/17{lock_tag}",
                            (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(disp, f"FPS {fps_avg:.0f}  Joints {n_j}/17{lock_tag}",
                            (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if pose.is_locked else WHITE, 2)

                if anom:
                    disp_label = action_label if action_label else "(idle)"
                    is_threat = disp_label in THREAT_ACTIONS
                    ac = RED if is_threat else (GRAY_DIM if disp_label == "(idle)" else CYAN)
                    conf_str = f" ({action_conf:.0%})" if action_conf > 0 else ""
                    atxt = f"Action: {disp_label}{conf_str}"
                    cv2.putText(disp, atxt, (8,52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                    cv2.putText(disp, atxt, (8,52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ac, 2)
                    if is_threat and action_conf >= 0.45:
                        cv2.putText(disp, "THREAT", (8, 78),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                        cv2.putText(disp, "THREAT", (8, 78),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

                if ae_mse is not None:
                    col = RED if anom else GREEN
                    lbl = "ANOMALY" if anom else "normal"
                    cv2.putText(disp, f"AE {lbl}  {ae_mse:.4f}", (8,H-65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 3)
                    cv2.putText(disp, f"AE {lbl}  {ae_mse:.4f}", (8,H-65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)

                # Bottom bar
                overlay = disp.copy()
                cv2.rectangle(overlay, (0, H-50), (W, H), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.5, disp, 0.5, 0, disp)

                _fill_rounded(disp, (10, H-45), (105, H-10), BG_CARD, 10)
                _rounded_rect_border(disp, (10, H-45), (105, H-10), BLUE_DIM, 10, 1)
                cv2.putText(disp, "< HOME", (16, H-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 2, cv2.LINE_AA)

                lock_fill = (0, 100, 0) if pose.is_locked else BG_CARD
                lock_border = (0, 255, 0) if pose.is_locked else BLUE_DIM
                lx1, lx2 = W//2 - 50, W//2 + 50
                _fill_rounded(disp, (lx1, H-45), (lx2, H-10), lock_fill, 10)
                _rounded_rect_border(disp, (lx1, H-45), (lx2, H-10), lock_border, 10, 1)
                lock_txt = "UNLOCK" if pose.is_locked else "LOCK"
                cv2.putText(disp, lock_txt, (lx1+10, H-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 2, cv2.LINE_AA)

                rec_fill = (40, 30, 180) if recording else BG_CARD
                _fill_rounded(disp, (W-115, H-45), (W-10, H-10), rec_fill, 10)
                _rounded_rect_border(disp, (W-115, H-45), (W-10, H-10), RED if recording else BLUE_DIM, 10, 1)
                cv2.putText(disp, "STOP" if recording else "REC", (W-95, H-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)

                if recording:
                    elapsed = time.time() - rec_start
                    mins, secs = int(elapsed//60), int(elapsed%60)
                    if int(elapsed*2)%2==0:
                        cv2.circle(disp, (W-25, 22), 8, RED, -1)
                    cv2.putText(disp, f"REC {mins:02d}:{secs:02d}", (W-150, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
                    if video_writer:
                        video_writer.write(disp)

                cv2.imshow(WIN, disp)
                cv2.waitKey(1)

                click = pop_click()
                if click:
                    cx, cy = click
                    if in_rect(cx, cy, 10, H-45, 105, H-10):
                        if recording and video_writer:
                            video_writer.release(); video_writer = None; recording = False
                            print(f"  [REC] Stopped -> {rec_path.name}")
                        mode = MODE_HOME; print("[MODE] -> Home")
                    elif in_rect(cx, cy, W//2-50, H-45, W//2+50, H-10):
                        if pose.is_locked:
                            pose.unlock()
                            print("  [LOCK] Unlocked - tracking nearest person")
                        else:
                            pose.lock_on(depth_frame=raw)
                            print("  [LOCK] Locked onto current target")
                    elif in_rect(cx, cy, W-115, H-45, W-10, H-10):
                        if not recording:
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            rec_path = clips_dir / f"clip_{ts}.mp4"
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(str(rec_path), fourcc, 15.0, (W, H))
                            rec_start = time.time(); recording = True
                            print(f"  [REC] Recording -> {rec_path.name}")
                        else:
                            recording = False
                            if video_writer: video_writer.release(); video_writer = None
                            print(f"  [REC] Stopped ({time.time()-rec_start:.1f}s) -> {rec_path.name}")
                    else:
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        sp = screenshots_dir / f"ss_{ts}.png"
                        cv2.imwrite(str(sp), disp)
                        print(f"  [SCREENSHOT] {sp.name}")
                continue

            # ── GALLERY ───────────────────────────────────────────────────
            if mode == MODE_GALLERY:
                gal_img, gallery_rects, gallery_total_pages, gallery_del_rects = \
                    draw_gallery(gallery_clips, gallery_page, delete_confirm_idx)
                cv2.imshow(WIN, gal_img)
                cv2.waitKey(30)

                click = pop_click()
                if click:
                    cx, cy = click
                    if in_rect(cx, cy, 10, 12, 110, 48):
                        mode = MODE_HOME; delete_confirm_idx = -1
                        print("[MODE] -> Home")
                    elif gallery_page > 0 and in_rect(cx, cy, 70, H-48, 170, H-16):
                        gallery_page -= 1; delete_confirm_idx = -1
                    elif gallery_page < gallery_total_pages-1 and in_rect(cx, cy, W-170, H-48, W-70, H-16):
                        gallery_page += 1; delete_confirm_idx = -1
                    else:
                        handled = False
                        for (dx1, dy1, dx2, dy2, didx) in gallery_del_rects:
                            if in_rect(cx, cy, dx1-5, dy1-5, dx2+5, dy2+5):
                                if delete_confirm_idx == didx:
                                    clip_path = gallery_clips[didx][0]
                                    print(f"  [DELETE] {clip_path.name}")
                                    clip_path.unlink(missing_ok=True)
                                    gallery_clips = get_clip_list(clips_dir)
                                    if gallery_page > 0 and gallery_page * 6 >= len(gallery_clips):
                                        gallery_page -= 1
                                    delete_confirm_idx = -1
                                else:
                                    delete_confirm_idx = didx
                                handled = True; break
                        if not handled:
                            for (x1, y1, x2, y2, idx) in gallery_rects:
                                if in_rect(cx, cy, x1, y1, x2, y2):
                                    clip_path = gallery_clips[idx][0]
                                    play_cap = cv2.VideoCapture(str(clip_path))
                                    if play_cap.isOpened():
                                        play_fps = play_cap.get(cv2.CAP_PROP_FPS) or 15
                                        fc = play_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                                        play_total_sec = fc / max(play_fps, 1)
                                        play_clip_name = clip_path.stem
                                        speed_idx = 1; play_fail_count = 0; mode = MODE_PLAY
                                        delete_confirm_idx = -1
                                        print(f"[PLAY] {clip_path.name}")
                                    else:
                                        play_cap = None
                                    break
                            else:
                                delete_confirm_idx = -1
                continue

            # ── PLAYBACK ──────────────────────────────────────────────────
            if mode == MODE_PLAY:
                got_frame = False
                try:
                    if play_cap and play_cap.isOpened():
                        ret, frame = play_cap.read()
                        if ret:
                            frame = cv2.resize(frame, (W, H))
                            pos_sec = play_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            play_btns = draw_play_hud(frame, pos_sec, play_total_sec,
                                                      speed_idx, speeds, play_clip_name)
                            cv2.imshow(WIN, frame)
                            got_frame = True
                            play_fail_count = 0
                        else:
                            play_fail_count += 1
                            if play_fail_count > 3:
                                # Clip unplayable - bail to gallery
                                print(f"  [PLAY] Cannot read clip, returning to gallery")
                                play_cap.release(); play_cap = None
                                mode = MODE_GALLERY
                                gallery_clips = get_clip_list(clips_dir)
                                print("[MODE] -> Gallery")
                                continue
                            play_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        # No valid capture - go back
                        mode = MODE_GALLERY
                        gallery_clips = get_clip_list(clips_dir)
                        print("[MODE] -> Gallery (cap invalid)")
                        continue
                except Exception as e:
                    print(f"  [PLAY] Error: {e}")
                    if play_cap:
                        try: play_cap.release()
                        except Exception: pass
                        play_cap = None
                    mode = MODE_GALLERY
                    gallery_clips = get_clip_list(clips_dir)
                    continue

                delay = max(1, int((1000.0 / max(play_fps, 1)) / speeds[speed_idx]))
                cv2.waitKey(delay)

                click = pop_click()
                if click:
                    cx, cy = click
                    if play_btns.get('back') and in_rect(cx, cy, *play_btns['back']):
                        if play_cap: play_cap.release(); play_cap = None
                        mode = MODE_GALLERY
                        gallery_clips = get_clip_list(clips_dir)
                        print("[MODE] -> Gallery")
                    else:
                        for si in range(3):
                            key = f'speed_{si}'
                            if key in play_btns and in_rect(cx, cy, *play_btns[key]):
                                speed_idx = si
                                print(f"  [SPEED] {speeds[si]}x")
                                break
                continue

    finally:
        if video_writer: video_writer.release()
        if play_cap: play_cap.release()
        rs_pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Launcher stopped.")


if __name__ == '__main__':
    main()
