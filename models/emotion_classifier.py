"""
Lightweight facial emotion classifier for Third Eye Shield with few-shot personalization.

OPT-IN ONLY -- loaded only when user explicitly enables emotion detection.
Face crops processed in-memory only, never stored to disk or transmitted.

Base model: Mini-Xception (FER2013, 48x48 grayscale, ~60K params)
Personalization: Prototypical few-shot learning (per-user profile)

Two TFLite models:
  - emotion_fer2013.tflite:          Full classifier  (48x48x1 -> 7 probs)
  - emotion_fer2013_features.tflite: Feature extractor (48x48x1 -> 128-dim)

7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
"""
import os
import time
import numpy as np
import cv2
from pathlib import Path

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
NEGATIVE_EMOTIONS = {'angry', 'disgust', 'fear', 'sad'}
FEATURE_DIM = 128  # Mini-Xception module 4 output depth


class FaceDetector:
    """MediaPipe face detection wrapper -- lightweight SSD, ~5ms on RPi."""

    def __init__(self, min_detection_confidence=0.5):
        import mediapipe as mp
        self._face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range (<2m)
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, rgb_image):
        """
        Detect faces in an RGB image.

        Returns:
            list of (x1, y1, x2, y2) bounding boxes in pixel coordinates,
            sorted by area (largest first).  Empty list if no face found.
        """
        results = self._face_det.process(rgb_image)
        if not results.detections:
            return []

        h, w = rgb_image.shape[:2]
        boxes = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width) * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                boxes.append((x1, y1, x2, y2))

        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes

    def close(self):
        self._face_det.close()


class EmotionClassifier:
    """
    Facial emotion classifier with few-shot per-user personalization.

    Two classification modes:
      1. Base mode: TFLite classifier -> 7-class softmax (no profile)
      2. Few-shot mode: Feature extractor -> cosine similarity to user
         prototypes (when a profile is loaded)

    Few-shot workflow:
        clf = EmotionClassifier('models/emotion_fer2013.tflite',
                                'models/emotion_fer2013_features.tflite')
        clf.load_profile('UNCLE_TAN')
        label, conf, probs = clf.classify(face_crop_rgb)
    """

    DEFAULT_CLASSIFIER = 'models/emotion_fer2013.tflite'
    DEFAULT_FEATURES   = 'models/emotion_fer2013_features.tflite'
    DEFAULT_PROFILES   = 'data/emotion_profiles'

    def __init__(self, model_path=None, feature_model_path=None,
                 profiles_dir=None, conf_threshold=0.35):
        self.conf_threshold = conf_threshold
        self._classifier = None     # TFLite classifier interpreter
        self._feat_extractor = None # TFLite feature extractor interpreter
        self._profiles_dir = Path(profiles_dir or self.DEFAULT_PROFILES)

        # Per-user few-shot state
        self._prototypes = None       # (7, 128) mean feature per emotion
        self._prototype_counts = None # (7,) samples per emotion
        self._profile_features = None # (N, 128) all stored features
        self._profile_labels = None   # (N,) emotion index per feature
        self._profile_user_id = None

        # Legacy calibration (backward compat)
        self._calibration = None

        # Init classifier
        cls_path = model_path or self.DEFAULT_CLASSIFIER
        if os.path.isfile(cls_path):
            self._init_tflite_classifier(cls_path)
        else:
            print(f"[EMOTION] Classifier not found: {cls_path}")
            print("          Run: python3 scripts/setup_emotion.py")

        # Init feature extractor
        feat_path = feature_model_path or self.DEFAULT_FEATURES
        if os.path.isfile(feat_path):
            self._init_tflite_features(feat_path)
        elif self._classifier is not None:
            print("[EMOTION] Feature model not found -- few-shot disabled")
            print("          Run: python3 scripts/setup_emotion.py")

    # ---- TFLite init --------------------------------------------------

    def _init_tflite_classifier(self, path):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        self._cls_inp_idx = inp['index']
        self._cls_inp_shape = inp['shape']   # [1, 48, 48, 1]
        self._cls_out_idx = interp.get_output_details()[0]['index']
        self._classifier = interp
        print(f"[EMOTION] Classifier: {Path(path).name}  "
              f"input={list(self._cls_inp_shape)}")

    def _init_tflite_features(self, path):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        self._feat_inp_idx = inp['index']
        self._feat_inp_shape = inp['shape']  # [1, 48, 48, 1]
        out = interp.get_output_details()[0]
        self._feat_out_idx = out['index']
        self._feat_dim = out['shape'][-1]    # 128
        self._feat_extractor = interp
        print(f"[EMOTION] Feature extractor: {Path(path).name}  "
              f"dim={self._feat_dim}")

    # ---- Preprocessing ------------------------------------------------

    def _preprocess(self, face_crop_rgb, target_shape):
        """Convert RGB face crop to normalized grayscale tensor."""
        h, w = target_shape[1], target_shape[2]
        gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (w, h))
        inp = gray.astype(np.float32) / 255.0
        return inp.reshape(target_shape)

    # ---- Feature extraction -------------------------------------------

    def extract_features(self, face_crop_rgb):
        """
        Extract 128-dim embedding from a face crop.

        Args:
            face_crop_rgb: RGB image of a face (any size).

        Returns:
            np.ndarray of shape (128,), or None if extractor unavailable.
        """
        if self._feat_extractor is None:
            return None
        inp = self._preprocess(face_crop_rgb, self._feat_inp_shape)
        self._feat_extractor.set_tensor(self._feat_inp_idx, inp)
        self._feat_extractor.invoke()
        features = self._feat_extractor.get_tensor(self._feat_out_idx)[0]
        return features.astype(np.float64)

    # ---- Base classification ------------------------------------------

    def _classify_base(self, face_crop_rgb):
        """Run base 7-class classifier (no personalization)."""
        if self._classifier is None:
            return 'neutral', 0.0, dict(zip(EMOTION_LABELS,
                                            [0]*6 + [1.0]))
        inp = self._preprocess(face_crop_rgb, self._cls_inp_shape)
        self._classifier.set_tensor(self._cls_inp_idx, inp)
        self._classifier.invoke()
        logits = self._classifier.get_tensor(
            self._cls_out_idx)[0].astype(np.float64)

        if self._calibration is not None:
            logits = logits + self._calibration

        probs = _softmax(logits)
        return self._decode(probs)

    # ---- Few-shot classification --------------------------------------

    def _classify_fewshot(self, face_crop_rgb):
        """Classify using cosine similarity to stored per-user prototypes."""
        features = self.extract_features(face_crop_rgb)
        if features is None:
            return self._classify_base(face_crop_rgb)

        # L2-normalize
        feat_norm = features / (np.linalg.norm(features) + 1e-8)
        proto_norms = self._prototypes / (
            np.linalg.norm(self._prototypes, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        similarities = proto_norms @ feat_norm

        # Mask emotions with no samples
        valid = self._prototype_counts > 0
        if not np.any(valid):
            return self._classify_base(face_crop_rgb)

        # Also run base model and blend
        base_label, base_conf, base_probs = self._classify_base(face_crop_rgb)
        base_p = np.array([base_probs[e] for e in EMOTION_LABELS])

        # Temperature-scaled softmax of similarities
        temperature = 10.0
        masked_sim = np.full(len(EMOTION_LABELS), -1e9)
        masked_sim[valid] = similarities[valid] * temperature
        fewshot_p = _softmax(masked_sim)

        # Blend: 60% few-shot, 40% base (base regularizes predictions)
        blended = 0.6 * fewshot_p + 0.4 * base_p
        blended = blended / (blended.sum() + 1e-8)

        return self._decode(blended)

    # ---- Public classify API ------------------------------------------

    def classify(self, face_crop_rgb):
        """
        Classify emotion from an RGB face crop.

        Uses few-shot if a profile is loaded, otherwise base classifier.

        Returns:
            (label: str, confidence: float, probs: dict)
        """
        if self._prototypes is not None:
            return self._classify_fewshot(face_crop_rgb)
        return self._classify_base(face_crop_rgb)

    def _decode(self, probs):
        prob_dict = dict(zip(EMOTION_LABELS, probs.tolist()))
        best_idx = int(np.argmax(probs))
        label = EMOTION_LABELS[best_idx]
        conf = float(probs[best_idx])
        if conf < self.conf_threshold:
            label = 'neutral'
            conf = float(probs[EMOTION_LABELS.index('neutral')])
        return label, conf, prob_dict

    # ---- Profile management (few-shot) --------------------------------

    def load_profile(self, user_id):
        """
        Load a per-user emotion profile for few-shot classification.

        Profiles are stored as .npz files in profiles_dir.
        """
        path = self._profiles_dir / f"{user_id}.npz"
        if not path.exists():
            print(f"[EMOTION] Profile not found: {path}")
            return False

        data = np.load(str(path), allow_pickle=True)
        self._profile_features = data['features']   # (N, 128)
        self._profile_labels = data['labels']        # (N,)
        self._profile_user_id = str(data.get('user_id', user_id))
        self._recompute_prototypes()
        n = len(self._profile_labels)
        n_emo = int(np.sum(self._prototype_counts > 0))
        print(f"[EMOTION] Profile loaded: {user_id} "
              f"({n} samples, {n_emo}/7 emotions)")
        return True

    def save_profile(self, user_id=None):
        """Save current profile to disk."""
        uid = user_id or self._profile_user_id
        if uid is None or self._profile_features is None:
            return False

        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        path = self._profiles_dir / f"{uid}.npz"
        np.savez(str(path),
                 features=self._profile_features,
                 labels=self._profile_labels,
                 user_id=uid)
        print(f"[EMOTION] Profile saved: {path}")
        return True

    def register_sample(self, user_id, emotion_idx, face_crop_rgb):
        """
        Add a face sample to a user's profile for few-shot learning.

        Args:
            user_id: User identifier string.
            emotion_idx: Index into EMOTION_LABELS (0-6).
            face_crop_rgb: RGB face crop image.

        Returns:
            True if sample was successfully registered.
        """
        features = self.extract_features(face_crop_rgb)
        if features is None:
            return False

        self._profile_user_id = user_id

        if self._profile_features is None:
            self._profile_features = features.reshape(1, -1)
            self._profile_labels = np.array([emotion_idx])
        else:
            self._profile_features = np.vstack([
                self._profile_features, features.reshape(1, -1)])
            self._profile_labels = np.append(
                self._profile_labels, emotion_idx)

        self._recompute_prototypes()
        return True

    def get_profile_stats(self):
        """Get sample counts per emotion for current profile."""
        if self._prototype_counts is None:
            return {}
        return {EMOTION_LABELS[i]: int(self._prototype_counts[i])
                for i in range(len(EMOTION_LABELS))}

    def _recompute_prototypes(self):
        """Recompute prototype centroids from stored features."""
        n_classes = len(EMOTION_LABELS)
        self._prototypes = np.zeros((n_classes, FEATURE_DIM), dtype=np.float64)
        self._prototype_counts = np.zeros(n_classes, dtype=np.int32)

        for i in range(n_classes):
            mask = self._profile_labels == i
            count = int(mask.sum())
            self._prototype_counts[i] = count
            if count > 0:
                self._prototypes[i] = self._profile_features[mask].mean(axis=0)

    # ---- Legacy compat ------------------------------------------------

    def load_calibration(self, calibration_path):
        """
        Load legacy calibration (logit offset vector).
        For backward compatibility with old .npy calibration files.
        """
        if os.path.isfile(calibration_path):
            ext = os.path.splitext(calibration_path)[1]
            if ext == '.npz':
                # New-style profile
                user_id = Path(calibration_path).stem
                return self.load_profile(user_id)
            else:
                self._calibration = np.load(calibration_path)


class EmotionTracker:
    """
    Track emotion over time with a rolling window.
    Reports sustained negative emotion as a concern signal.
    """

    def __init__(self, window_sec=300, negative_ratio_threshold=0.6):
        self.window_sec = window_sec
        self.neg_ratio_thr = negative_ratio_threshold
        self._history = []  # list of (timestamp, label)

    def update(self, label):
        now = time.time()
        self._history.append((now, label))
        cutoff = now - self.window_sec
        self._history = [(t, l) for t, l in self._history if t >= cutoff]

    @property
    def is_prolonged_negative(self):
        if len(self._history) < 5:
            return False
        neg_count = sum(1 for _, l in self._history if l in NEGATIVE_EMOTIONS)
        return (neg_count / len(self._history)) >= self.neg_ratio_thr

    @property
    def dominant_emotion(self):
        if not self._history:
            return 'neutral'
        from collections import Counter
        counts = Counter(l for _, l in self._history)
        return counts.most_common(1)[0][0]

    def reset(self):
        self._history.clear()


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-8)
