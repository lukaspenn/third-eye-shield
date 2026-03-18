"""Shared skeleton constants and drawing utilities for MoveNet 17-joint skeletons."""
import numpy as np

# MoveNet 17-joint bone connections for visualization
MOVENET_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (0, 5), (0, 6),                          # shoulders to nose
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 6),                                  # shoulder bridge
    (5, 11), (6, 12),                        # torso sides
    (11, 12),                                # hip bridge
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

# Per-joint confidence threshold
JOINT_CONF_THR = 0.25

# Threat action labels
THREAT_ACTIONS = {'punching/slapping', 'pushing other person', 'kicking something', 'falling'}


class JointSmoother:
    """Lightweight per-joint EMA to reduce MoveNet quantisation jitter.

    Only smooths x, y; confidence is passed through unchanged.
    alpha=1.0 means no smoothing (raw), lower = more smoothing.
    """

    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha
        self._prev = None

    def __call__(self, kps: np.ndarray) -> np.ndarray:
        """kps: (17, 3) [x_px, y_px, conf] -> smoothed copy."""
        out = kps.copy()
        if self._prev is None:
            self._prev = kps[:, :2].copy()
        else:
            out[:, :2] = self.alpha * kps[:, :2] + (1 - self.alpha) * self._prev
            visible = kps[:, 2] > JOINT_CONF_THR
            self._prev[visible] = out[visible, :2]
        return out

    def reset(self):
        self._prev = None


def draw_skeleton(img, keypoints, bone_color=(255, 255, 255), joint_color=(0, 0, 255),
                  bone_thickness=3, shadow_thickness=7):
    """Draw MoveNet 17-joint skeleton on an image.

    Args:
        img: BGR image to draw on (modified in-place).
        keypoints: (17, 3) array [x_px, y_px, conf].
        bone_color: BGR colour for bone lines.
        joint_color: BGR colour for joint circles.
        bone_thickness: Line width for bones.
        shadow_thickness: Line width for shadow behind bones.
    """
    import cv2
    ih, iw = img.shape[:2]

    def px(i):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        return (max(0, min(x, iw - 1)), max(0, min(y, ih - 1)))

    for a, b in MOVENET_BONES:
        if keypoints[a, 2] > JOINT_CONF_THR and keypoints[b, 2] > JOINT_CONF_THR:
            cv2.line(img, px(a), px(b), (0, 0, 0), shadow_thickness)
            cv2.line(img, px(a), px(b), bone_color, bone_thickness)
    for j in range(17):
        if keypoints[j, 2] > JOINT_CONF_THR:
            p = px(j)
            r = 8 if j == 0 else 5
            cv2.circle(img, p, r + 2, (0, 0, 0), -1)
            cv2.circle(img, p, r, joint_color, -1)
