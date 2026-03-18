"""
Wellness feature extraction from MoveNet 17-joint skeletons.

Provides posture scoring, sedentary tracking, and wellness level
computation for Third Eye Shield elderly wellness monitoring.

MoveNet joint layout (17 joints):
    0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear,
    5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
    9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip,
    13=L_knee, 14=R_knee, 15=L_ankle, 16=R_ankle
"""
import time
import numpy as np

# Joint indices (MoveNet 17)
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

# Wellness levels
WELLNESS_ACTIVE   = 0   # exercising / high motion
WELLNESS_NORMAL   = 1   # standard daily activity
WELLNESS_SEDENTARY = 2  # extended inactivity
WELLNESS_CONCERN  = 3   # behavioural anomaly (posture, emotion)
WELLNESS_ALERT    = 4   # emergency (fall, prolonged floor time)

WELLNESS_NAMES = {
    0: "Active",
    1: "Normal",
    2: "Sedentary",
    3: "Concern",
    4: "Alert",
}

WELLNESS_COLORS_BGR = {
    0: (0, 200, 0),      # green
    1: (200, 200, 0),    # cyan-ish
    2: (0, 180, 255),    # orange
    3: (0, 100, 255),    # dark orange
    4: (0, 0, 255),      # red
}

# Actions that indicate active exercise
ACTIVE_ACTIONS = {'clapping', 'arm circles', 'kicking something', 'sit down and up'}

# Actions that are alerts
ALERT_ACTIONS = {'falling'}

# Posture quality thresholds
_MIN_CONF = 0.25  # joint confidence threshold for posture calc


def compute_posture_score(skeleton):
    """
    Compute a 0-100 posture quality score from a single MoveNet (17,3) skeleton.

    Evaluates:
        - Shoulder alignment: tilt of shoulder line from horizontal
        - Head drop: how much the nose drops below mid-shoulder level
        - Spine angle: forward lean from mid-hip → mid-shoulder → nose

    Args:
        skeleton: (17, 3) array [x_px, y_px, confidence] or (17, 2) [x_norm, y_norm]

    Returns:
        float in [0, 100] or None if insufficient joints visible.
    """
    kps = np.asarray(skeleton, dtype=np.float32)
    has_conf = kps.shape[1] >= 3

    def visible(idx):
        if not has_conf:
            return True
        return kps[idx, 2] > _MIN_CONF

    # Need shoulders, hips, and nose
    if not (visible(L_SHOULDER) and visible(R_SHOULDER) and
            visible(L_HIP) and visible(R_HIP) and visible(NOSE)):
        return None

    xy = kps[:, :2]

    # 1. Shoulder tilt: angle from horizontal (0 = perfect, 90 = worst)
    ls, rs = xy[L_SHOULDER], xy[R_SHOULDER]
    dx = rs[0] - ls[0]
    dy = rs[1] - ls[1]
    shoulder_tilt_deg = abs(np.degrees(np.arctan2(dy, dx + 1e-8)))
    # Score: 0° tilt → 100, 30°+ tilt → 0
    shoulder_score = max(0.0, 100.0 - shoulder_tilt_deg * (100.0 / 30.0))

    # 2. Head drop: vertical distance from nose to mid-shoulder
    #    (in image coords, y increases downward, so positive = nose below shoulders)
    mid_shoulder = (ls + rs) / 2.0
    head_drop = xy[NOSE][1] - mid_shoulder[1]
    # Normalise by shoulder width (scale-invariant)
    shoulder_width = max(np.linalg.norm(rs - ls), 1e-6)
    head_drop_ratio = head_drop / shoulder_width
    # Ideal: nose is above shoulders (negative ratio). Excessive drop = poor.
    # Score: ratio <= -0.3 → 100, ratio >= 0.5 → 0
    head_score = np.clip((0.5 - head_drop_ratio) / 0.8 * 100.0, 0.0, 100.0)

    # 3. Spine lean: angle at mid-shoulder in the chain mid-hip → mid-shoulder → nose
    mid_hip = (xy[L_HIP] + xy[R_HIP]) / 2.0
    v_lower = mid_hip - mid_shoulder       # hip → shoulder vector
    v_upper = xy[NOSE] - mid_shoulder      # shoulder → nose vector
    cos_angle = np.dot(v_lower, v_upper) / (
        np.linalg.norm(v_lower) * np.linalg.norm(v_upper) + 1e-8)
    spine_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    # Ideal: ~170-180° (straight line). < 140° = significant lean.
    spine_score = np.clip((spine_angle - 100.0) / 80.0 * 100.0, 0.0, 100.0)

    # Weighted average
    score = 0.30 * shoulder_score + 0.30 * head_score + 0.40 * spine_score
    return float(np.clip(score, 0.0, 100.0))


class PostureTracker:
    """EMA-smoothed posture score tracker."""

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._score = None

    def update(self, skeleton):
        raw = compute_posture_score(skeleton)
        if raw is None:
            return self._score  # keep previous
        if self._score is None:
            self._score = raw
        else:
            self._score = self.alpha * raw + (1 - self.alpha) * self._score
        return self._score

    @property
    def score(self):
        return self._score

    def reset(self):
        self._score = None


class SedentaryTracker:
    """Track time since last non-sedentary activity."""

    def __init__(self, threshold_sec=1800):
        self.threshold_sec = threshold_sec  # 30 min default
        self._last_active_t = time.time()
        self._sedentary = False

    def update(self, action_label, motion_energy=0.0):
        """
        Call every frame with current action label and optional motion energy.
        Returns sedentary duration in seconds.
        """
        now = time.time()
        is_active = (
            action_label in ACTIVE_ACTIONS or
            motion_energy > 0.02  # significant overall motion
        )
        if is_active:
            self._last_active_t = now
            self._sedentary = False

        duration = now - self._last_active_t
        self._sedentary = duration >= self.threshold_sec
        return duration

    @property
    def is_sedentary(self):
        return self._sedentary

    @property
    def sedentary_minutes(self):
        return (time.time() - self._last_active_t) / 60.0

    def reset(self):
        self._last_active_t = time.time()
        self._sedentary = False


def compute_wellness_level(action_label, posture_score, sedentary_tracker,
                           emotion_label=None, emotion_enabled=False):
    """
    Determine the current wellness level (0-4) from all available signals.

    Args:
        action_label: str from action classifier (e.g. "falling", "arm circles")
        posture_score: float 0-100 or None
        sedentary_tracker: SedentaryTracker instance
        emotion_label: str or None (only used if emotion_enabled)
        emotion_enabled: bool - whether emotion module is active

    Returns:
        (level: int, reason: str)
    """
    # Level 4: Alert — fall detected
    if action_label in ALERT_ACTIONS:
        return WELLNESS_ALERT, f"Fall detected: {action_label}"

    # Level 3: Concern — poor posture or prolonged negative emotion
    concerns = []
    if posture_score is not None and posture_score < 35:
        concerns.append(f"poor posture ({posture_score:.0f}/100)")
    if emotion_enabled and emotion_label in ('angry', 'fear', 'sad', 'disgust'):
        concerns.append(f"negative emotion: {emotion_label}")
    if concerns:
        return WELLNESS_CONCERN, "; ".join(concerns)

    # Level 2: Sedentary
    if sedentary_tracker.is_sedentary:
        mins = sedentary_tracker.sedentary_minutes
        return WELLNESS_SEDENTARY, f"inactive {mins:.0f} min"

    # Level 0: Active — exercising
    if action_label in ACTIVE_ACTIONS:
        return WELLNESS_ACTIVE, f"exercise: {action_label}"

    # Level 1: Normal
    return WELLNESS_NORMAL, "normal activity"
