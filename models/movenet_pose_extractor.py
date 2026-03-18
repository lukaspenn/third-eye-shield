#!/usr/bin/env python3
"""
MoveNet single-pose extractor with depth-gated nearest-person ROI.

Strategy for crowded scenes:
  1. Use aligned depth frame to find the nearest person-sized blob
  2. Crop RGB to that region (padded)
  3. Run fast single-pose MoveNet Lightning int8 (192x192) on the crop
  4. Map keypoints back to full-frame coordinates

This gives ~20 FPS on Raspberry Pi 4 (vs 1-2 FPS with multipose@320)
while reliably isolating the person closest to the camera.
"""
import numpy as np
import cv2
import importlib
import tensorflow as tf
from pathlib import Path

# MoveNet keypoint connections for visualization
KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (0, 5), (0, 6), (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 6), (5, 11), (6, 12),  # Torso
    (11, 12), (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)  # Right leg
]

# MoveNet keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def _load_tflite_interpreter(model_path: str, delegate: str = 'cpu'):
    """Load a TFLite interpreter with optional EdgeTPU delegate."""
    import tensorflow as tf_module
    if delegate == 'edgetpu':
        try:
            tflite_rt = importlib.import_module('tflite_runtime.interpreter')
            load_delegate = getattr(tflite_rt, 'load_delegate')
            Interpreter = getattr(tflite_rt, 'Interpreter')
            return Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
        except Exception as e_rt:
            print(f"[MoveNet] EdgeTPU delegate (tflite_runtime) failed, trying TF Lite: {e_rt}")
            try:
                load_delegate_tf = tf_module.lite.experimental.load_delegate
                return tf_module.lite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[load_delegate_tf('libedgetpu.so.1')]
                )
            except Exception as e_tf:
                print(f"[MoveNet] EdgeTPU delegate (tf.lite) failed, falling back to CPU: {e_tf}")
    return tf_module.lite.Interpreter(model_path=model_path)


# ======================================================================
# Depth-based nearest-person ROI extraction
# ======================================================================

def find_nearest_person_roi(depth_frame_np, min_depth_mm=300, max_depth_mm=3000,
                            min_blob_area=3000, pad_ratio=0.35):
    """Find bounding box of the nearest person-sized blob in a depth image.

    Args:
        depth_frame_np: uint16 depth image in millimetres (H, W).
        min_depth_mm:   ignore pixels closer than this (sensor noise).
        max_depth_mm:   ignore pixels farther than this.
        min_blob_area:  minimum contour area (px) to count as a person.
        pad_ratio:      fractional padding added around the blob bbox.

    Returns:
        (x1, y1, x2, y2) in pixel coords, or None if no person found.
    """
    blobs = _find_person_blobs(depth_frame_np, min_depth_mm, max_depth_mm,
                               min_blob_area, pad_ratio)
    if not blobs:
        return None
    # Pick nearest (smallest median depth)
    blobs.sort(key=lambda b: b[1])
    return blobs[0][0]


def _find_person_blobs(depth_frame_np, min_depth_mm=300, max_depth_mm=3000,
                       min_blob_area=3000, pad_ratio=0.35):
    """Return list of (padded_bbox, median_depth, center_xy) for all person-sized blobs."""
    h, w = depth_frame_np.shape[:2]

    mask = ((depth_frame_np > min_depth_mm) &
            (depth_frame_np < max_depth_mm)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    blobs = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_blob_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        roi_depth = depth_frame_np[by:by+bh, bx:bx+bw]
        valid = roi_depth[(roi_depth > min_depth_mm) & (roi_depth < max_depth_mm)]
        if len(valid) == 0:
            continue
        med = float(np.median(valid))
        # Padded bbox
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        blobs.append(((x1, y1, x2, y2), med, (cx, cy)))
    return blobs


# ======================================================================
# MoveNet Pose Extractor  (single-pose + depth-ROI)
# ======================================================================

class MoveNetPoseExtractor:
    """
    Fast single-pose MoveNet Lightning with depth-gated ROI cropping.

    When a depth frame is provided to extract(), the extractor:
      1. Finds the nearest person-sized blob via depth
      2. Crops the RGB to that region
      3. Runs single-pose MoveNet on the crop (~20 FPS on Pi 4)
      4. Maps keypoints back to full-frame coordinates

    Without a depth frame it runs on the full image (backward compat).
    """

    def __init__(self, model_path='models/movenet_lightning_int8.tflite',
                 conf_threshold=0.2, delegate: str = 'cpu', **_kwargs):
        self.conf_threshold = conf_threshold

        self.interpreter = _load_tflite_interpreter(model_path=model_path, delegate=delegate)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]  # 192 for Lightning int8

        # ROI tracking - EMA-smooth the crop window to avoid jitter
        self._prev_roi = None
        self._roi_alpha = 0.5
        self._roi_lost = 0
        self._ROI_PATIENCE = 8

        # Lock-on state
        self._locked = False
        self._lock_depth = None   # median depth (mm) of locked person
        self._lock_center = None  # (cx, cy) pixel centre of locked person
        self._LOCK_DEPTH_TOL = 700   # mm tolerance for depth matching
        self._LOCK_SPATIAL_TOL = 350  # px tolerance for centre matching
        self._lock_lost = 0
        self._LOCK_LOST_MAX = 120  # frames (~6s at 20fps) before auto-unlock
        self._LOCK_REF_ALPHA = 0.2  # EMA weight for updating lock reference

        print(f"[MoveNet] Loaded: {model_path}")
        print(f"[MoveNet] Mode: SinglePose + DepthROI + LockOn  |  {self.input_size}x{self.input_size}  |  conf>={conf_threshold}")

    # ------------------------------------------------------------------
    #  Core inference
    # ------------------------------------------------------------------
    def _infer(self, rgb_image):
        """Run MoveNet on an RGB image and return raw output tensor."""
        inp = cv2.resize(rgb_image, (self.input_size, self.input_size))
        inp = np.expand_dims(inp, axis=0)
        if self.input_details[0]['dtype'] == np.uint8:
            inp = inp.astype(np.uint8)
        else:
            inp = inp.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def _parse_keypoints(self, raw_output, h, w):
        """Parse single-pose output [1,1,17,3] -> (17,3) [x_px, y_px, conf]."""
        kps = raw_output[0, 0, :, :]  # (17, 3) - [y_norm, x_norm, conf]
        skeleton = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            y_n, x_n, c = kps[i]
            skeleton[i] = [x_n * w, y_n * h, c]
        return skeleton

    # ------------------------------------------------------------------
    #  Depth-ROI helpers
    # ------------------------------------------------------------------
    def _smooth_roi(self, new_roi):
        """EMA-smooth the ROI to prevent crop jitter."""
        if self._prev_roi is None:
            self._prev_roi = new_roi
            return new_roi
        a = self._roi_alpha
        smoothed = tuple(int(a * n + (1 - a) * p)
                         for n, p in zip(new_roi, self._prev_roi))
        self._prev_roi = smoothed
        return smoothed

    def _get_roi(self, depth_np):
        """Get current ROI from depth.  When locked, prefer the blob matching
        the locked person's depth+position; otherwise pick nearest."""
        blobs = _find_person_blobs(depth_np)
        if not blobs:
            self._roi_lost += 1
            if self._locked:
                self._lock_lost += 1
                if self._lock_lost > self._LOCK_LOST_MAX:
                    self.unlock()
                    print("[MoveNet] Auto-unlocked (person lost)")
            if self._prev_roi is not None and self._roi_lost <= self._ROI_PATIENCE:
                return self._prev_roi
            return None

        if self._locked and self._lock_depth is not None:
            # Find blob closest to locked person (combined depth+spatial score)
            best = None
            best_score = 1e9
            for bbox, med_depth, center in blobs:
                d_depth = abs(med_depth - self._lock_depth)
                d_spatial = ((center[0] - self._lock_center[0])**2 +
                             (center[1] - self._lock_center[1])**2) ** 0.5
                # Soft penalty: no hard gate - depth shift just increases score
                score = (d_depth / self._LOCK_DEPTH_TOL +
                         d_spatial / max(self._LOCK_SPATIAL_TOL, 1))
                if score < best_score:
                    best_score = score
                    best = (bbox, med_depth, center)
            # Accept if combined score is reasonable (< 3.0 allows generous match)
            if best is not None and best_score < 3.0:
                roi = self._smooth_roi(best[0])
                # EMA-smooth the lock reference to avoid noisy jumps
                a = self._LOCK_REF_ALPHA
                self._lock_depth = a * best[1] + (1 - a) * self._lock_depth
                self._lock_center = (
                    a * best[2][0] + (1 - a) * self._lock_center[0],
                    a * best[2][1] + (1 - a) * self._lock_center[1])
                self._roi_lost = 0
                self._lock_lost = 0
                return roi
            # Locked person not found among blobs
            self._lock_lost += 1
            self._roi_lost += 1
            if self._lock_lost > self._LOCK_LOST_MAX:
                self.unlock()
                print("[MoveNet] Auto-unlocked (person lost)")
            if self._prev_roi is not None and self._roi_lost <= self._ROI_PATIENCE:
                return self._prev_roi
            return None
        else:
            # Not locked - pick nearest blob
            blobs.sort(key=lambda b: b[1])
            nearest = blobs[0]
            roi = self._smooth_roi(nearest[0])
            self._roi_lost = 0
            return roi

    # ------------------------------------------------------------------
    #  Lock-on API
    # ------------------------------------------------------------------
    @property
    def is_locked(self):
        """True when tracking is locked onto a specific person."""
        return self._locked

    def lock_on(self, depth_frame=None):
        """Lock onto the currently tracked person.
        Call this when user confirms the target. If a depth frame is given,
        locks to the current nearest blob. Otherwise locks to whatever
        _prev_roi was tracking."""
        if depth_frame is not None:
            blobs = _find_person_blobs(depth_frame)
            if blobs:
                # Pick the blob closest to current _prev_roi if we have one
                if self._prev_roi is not None:
                    pcx = (self._prev_roi[0] + self._prev_roi[2]) / 2.0
                    pcy = (self._prev_roi[1] + self._prev_roi[3]) / 2.0
                    blobs.sort(key=lambda b: ((b[2][0]-pcx)**2 + (b[2][1]-pcy)**2))
                else:
                    blobs.sort(key=lambda b: b[1])  # nearest
                chosen = blobs[0]
                self._lock_depth = chosen[1]
                self._lock_center = chosen[2]
                self._prev_roi = chosen[0]
                self._locked = True
                self._lock_lost = 0
                print(f"[MoveNet] LOCKED ON  depth={self._lock_depth:.0f}mm  center=({self._lock_center[0]:.0f},{self._lock_center[1]:.0f})")
                return True
        # Fallback: lock based on prev_roi without depth specifics
        if self._prev_roi is not None:
            self._locked = True
            self._lock_lost = 0
            if self._lock_center is None:
                self._lock_center = ((self._prev_roi[0]+self._prev_roi[2])/2.0,
                                     (self._prev_roi[1]+self._prev_roi[3])/2.0)
            print(f"[MoveNet] LOCKED ON (from ROI)")
            return True
        print("[MoveNet] Cannot lock - no person tracked yet")
        return False

    def unlock(self):
        """Unlock tracking - return to nearest-person mode."""
        self._locked = False
        self._lock_depth = None
        self._lock_center = None
        self._lock_lost = 0
        print("[MoveNet] UNLOCKED - tracking nearest person")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def extract(self, rgb_image, draw=False, depth_frame=None):
        """
        Extract pose from RGB, optionally using depth for nearest-person ROI.

        Args:
            rgb_image:   (H, W, 3) RGB image.
            draw:        If True, return annotated image with skeleton + ROI box.
            depth_frame: Optional uint16 depth image (H, W) in mm.

        Returns:
            skeleton:  (17, 3) array  [x, y, confidence] in full-frame coords.
            annotated: Annotated image if draw=True, else original.
            detected:  True if >= 5 confident joints found.
        """
        h, w = rgb_image.shape[:2]
        roi = None

        if depth_frame is not None:
            roi = self._get_roi(depth_frame)

        if roi is not None:
            x1, y1, x2, y2 = roi
            crop = rgb_image[y1:y2, x1:x2]
            if crop.size == 0:
                roi = None

        if roi is not None:
            ch, cw = crop.shape[:2]
            raw_out = self._infer(crop)
            skeleton = self._parse_keypoints(raw_out, ch, cw)
            # Map back to full-frame coordinates
            skeleton[:, 0] += x1
            skeleton[:, 1] += y1
        else:
            raw_out = self._infer(rgb_image)
            skeleton = self._parse_keypoints(raw_out, h, w)

        n_good = int(np.sum(skeleton[:, 2] > self.conf_threshold))
        detected = n_good >= 5

        annotated = rgb_image
        if draw:
            annotated = rgb_image.copy()
            if detected:
                self._draw_keypoints(annotated, skeleton)
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                # Green = locked, Cyan = acquiring
                box_col = (0, 255, 0) if self._locked else (255, 200, 0)
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), box_col, 2)
                if self._locked:
                    cv2.putText(annotated, "LOCKED", (rx1, ry1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return skeleton, annotated, detected

    def reset_tracking(self):
        """Reset ROI tracking state (e.g. when scene changes)."""
        self._prev_roi = None
        self._roi_lost = 0
        # Don't reset lock - user explicitly controls that

    # ------------------------------------------------------------------
    #  Drawing
    # ------------------------------------------------------------------
    def _draw_keypoints(self, image, skeleton, color=(0, 255, 0), thickness=2):
        for edge in KEYPOINT_EDGES:
            x1, y1, c1 = skeleton[edge[0]]
            x2, y2, c2 = skeleton[edge[1]]
            if c1 > self.conf_threshold and c2 > self.conf_threshold:
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        for i, (x, y, conf) in enumerate(skeleton):
            if conf > self.conf_threshold:
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
        return image

    def release(self):
        """Release resources (compatibility method)."""
        pass


def convert_movenet_to_mediapipe_format(movenet_skeleton):
    """
    Convert MoveNet 17-keypoint skeleton to MediaPipe 33-keypoint format.
    Maps overlapping keypoints; unmapped ones are zero.
    """
    mediapipe_skeleton = np.zeros((33, 3), dtype=np.float32)
    mapping = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12,
        7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24,
        13: 25, 14: 26, 15: 27, 16: 28,
    }
    for mn_idx, mp_idx in mapping.items():
        mediapipe_skeleton[mp_idx] = movenet_skeleton[mn_idx]
    return mediapipe_skeleton


if __name__ == "__main__":
    print("Testing MoveNet SinglePose + DepthROI extractor...")
    extractor = MoveNetPoseExtractor(
        model_path='models/movenet_lightning_int8.tflite',
        conf_threshold=0.2)

    # Full-frame test (no depth)
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    skel, ann, det = extractor.extract(dummy, draw=True)
    print(f"Full-frame:  shape={skel.shape}  detected={det}  avg_conf={np.mean(skel[:,2]):.3f}")

    # Depth-ROI test (dummy depth with a near blob in centre)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[150:400, 200:450] = 1200  # simulated person at 1.2 m
    skel2, ann2, det2 = extractor.extract(dummy, draw=True, depth_frame=depth)
    print(f"Depth-ROI:   shape={skel2.shape}  detected={det2}  avg_conf={np.mean(skel2[:,2]):.3f}")

    mp = convert_movenet_to_mediapipe_format(skel)
    print(f"MediaPipe format: {mp.shape}")

    extractor.release()
    print("[OK] Test complete")
