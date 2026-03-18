"""
Kinematic feature extraction for MoveNet 17-joint skeletons.

MoveNet joint layout (17 joints):
    0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear,
    5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
    9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip,
    13=L_knee, 14=R_knee, 15=L_ankle, 16=R_ankle

Input:  list of (17, 2) arrays  -  normalised (x, y) per frame
Output: 1-D feature vector for RandomForest / XGBoost

Feature vector composition (J=17, D=2):
    Position mean:    J*D = 34
    Position std:     J*D = 34
    Speed mean:       J   = 17
    Speed max:        J   = 17
    Speed std:        J   = 17
    Accel mean:       J   = 17
    Accel max:        J   = 17
    Pairwise dists:   2+2+2 = 6   (wrist-to-nose mean/min, wrist spread, ankle spread)
    Motion energy:    1
    ----------------------------------
    TOTAL:            160
"""
import numpy as np

# Semantic joint indices (MoveNet 17)
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Feature dimension constant (importable by other modules)
FEATURE_DIM = 160


def extract_kinematic_features(skel_sequence):
    """
    Extract statistical kinematic features from a temporal sequence of skeletons.
    Translation-invariant (centred on mid-hip).

    Args:
        skel_sequence: list/array of shape (F, 17, 2)  --  F frames,
                       17 MoveNet joints, 2 coordinates (normalised x, y).

    Returns:
        1-D np.float32 array of length FEATURE_DIM (160).
    """
    skel_seq = np.array(skel_sequence, dtype=np.float32)

    if skel_seq.ndim != 3:
        raise ValueError(f"Expected shape (F, 17, 2), got {skel_seq.shape}")

    F, J, D = skel_seq.shape

    # Fallback for extremely short clips
    if F < 3:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # ── 1. Translation invariance  (centre on mid-hip) ──────────────────
    mid_hip = (skel_seq[:, L_HIP:L_HIP+1, :] +
               skel_seq[:, R_HIP:R_HIP+1, :]) / 2.0          # (F, 1, D)
    centred = skel_seq - mid_hip

    # ── 2. Velocities / accelerations ───────────────────────────────────
    vel   = np.diff(centred, axis=0)                            # (F-1, J, D)
    speed = np.linalg.norm(vel, axis=-1)                        # (F-1, J)
    acc   = np.diff(vel, axis=0)                                # (F-2, J, D)
    acc_m = np.linalg.norm(acc, axis=-1)                        # (F-2, J)

    # ── 3. Aggregate statistics ─────────────────────────────────────────
    feats = []

    # Centred position statistics  (posture shape)
    feats.append(np.mean(centred, axis=0).ravel())              # J*D = 34
    feats.append(np.std(centred, axis=0).ravel())               # J*D = 34

    # Speed statistics  (how fast each joint moves)
    feats.append(np.mean(speed, axis=0))                        # J = 17
    feats.append(np.max(speed, axis=0))                         # J = 17
    feats.append(np.std(speed, axis=0))                         # J = 17

    # Acceleration statistics  (jerky / sharp motions)
    feats.append(np.mean(acc_m, axis=0))                        # J = 17
    feats.append(np.max(acc_m, axis=0))                         # J = 17

    # ── 4. Domain-specific pairwise distances ───────────────────────────
    # Wrist-to-nose  (drinking, phone, punching)
    l_wrist_nose = np.linalg.norm(skel_seq[:, L_WRIST] - skel_seq[:, NOSE], axis=-1)
    r_wrist_nose = np.linalg.norm(skel_seq[:, R_WRIST] - skel_seq[:, NOSE], axis=-1)
    feats.append(np.array([np.mean(l_wrist_nose), np.min(l_wrist_nose)]))   # 2
    feats.append(np.array([np.mean(r_wrist_nose), np.min(r_wrist_nose)]))   # 2

    # Spread of wrists and ankles  (arm circles, kicking, falling)
    wrist_spread = np.linalg.norm(skel_seq[:, L_WRIST] - skel_seq[:, R_WRIST], axis=-1)
    ankle_spread = np.linalg.norm(skel_seq[:, L_ANKLE] - skel_seq[:, R_ANKLE], axis=-1)
    feats.append(np.array([np.max(wrist_spread), np.max(ankle_spread)]))     # 2

    # ── 5. Global motion energy ─────────────────────────────────────────
    total_energy = np.sum(speed) / (F * J)
    feats.append(np.array([total_energy]))                                    # 1

    # ── Concatenate ─────────────────────────────────────────────────────
    out = np.concatenate(feats).astype(np.float32)
    np.nan_to_num(out, copy=False)

    assert out.shape[0] == FEATURE_DIM, \
        f"Feature dim mismatch: expected {FEATURE_DIM}, got {out.shape[0]}"
    return out
