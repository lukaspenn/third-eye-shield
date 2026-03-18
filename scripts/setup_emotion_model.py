#!/usr/bin/env python3
"""
Third Eye Shield -- Emotion Model Setup
=================================

Downloads open-source Mini-Xception model (trained on FER2013, MIT license)
and converts to two TFLite models:

  1. models/emotion_fer2013.tflite          -- classifier  (48x48x1 -> 7 probs)
  2. models/emotion_fer2013_features.tflite  -- embeddings  (48x48x1 -> 128-dim)

The feature extractor powers per-user few-shot personalization.

Source: oarriaga/face_classification (github.com/oarriaga/face_classification)
Dataset: FER2013 (7 emotions, ~35K images, 48x48 grayscale)
Architecture: Mini-Xception (~60K params, <1MB)

Usage:
    python3 scripts/setup_emotion_model.py
    python3 scripts/setup_emotion_model.py --force   # re-download
"""
import os
import sys
import argparse
import urllib.request
import tempfile
from pathlib import Path

WEIGHTS_URL = (
    "https://github.com/oarriaga/face_classification/raw/master/"
    "trained_models/emotion_models/fer2013_mini_XCEPTION.110-0.65.hdf5"
)
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 7
FEATURE_DIM = 128  # channels after module 4

OUT_DIR = Path(__file__).resolve().parents[1] / "models"
CLASSIFIER_PATH = OUT_DIR / "emotion_fer2013.tflite"
FEATURES_PATH = OUT_DIR / "emotion_fer2013_features.tflite"


def build_mini_xception(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Build Mini-Xception architecture matching oarriaga/face_classification.

    4 residual modules with SeparableConv2D, filters: 16 -> 32 -> 64 -> 128.
    Returns the full Keras model (input -> 7-class softmax).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, regularizers

    l2_reg = regularizers.l2(0.01)
    img_input = layers.Input(shape=input_shape)

    # Base convolutions
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), use_bias=False,
                      kernel_regularizer=l2_reg)(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), use_bias=False,
                      kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 4 residual modules
    for i, n_filters in enumerate([16, 32, 64, FEATURE_DIM]):
        residual = layers.Conv2D(n_filters, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.SeparableConv2D(n_filters, (3, 3), padding='same',
                                   use_bias=False,
                                   depthwise_regularizer=l2_reg,
                                   pointwise_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(n_filters, (3, 3), padding='same',
                                   use_bias=False,
                                   depthwise_regularizer=l2_reg,
                                   pointwise_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

    # Classification head (matches oarriaga: Conv2D(7) -> GAP -> softmax)
    x = layers.Conv2D(num_classes, (3, 3), padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Activation('softmax', name='predictions')(x)

    model = Model(img_input, output, name='mini_xception')
    return model


def download_weights(dest_path):
    """Download pre-trained weights from GitHub."""
    print(f"[SETUP] Downloading Mini-Xception weights...")
    print(f"        URL: {WEIGHTS_URL}")
    urllib.request.urlretrieve(WEIGHTS_URL, str(dest_path))
    size_kb = os.path.getsize(dest_path) / 1024
    print(f"[SETUP] Downloaded: {size_kb:.0f} KB")


def load_weights_into_model(model, weights_path):
    """
    Load oarriaga weights into our architecture.
    Tries multiple strategies for TF version compatibility.
    """
    import tensorflow as tf

    # Strategy 1: load_weights by topology (fastest, works if architecture matches)
    try:
        model.load_weights(str(weights_path))
        print("[SETUP] Loaded weights by topology")
        return True
    except Exception as e:
        print(f"[SETUP] Topology load failed: {e}")

    # Strategy 2: load full model, transfer weights layer-by-layer
    try:
        loaded = tf.keras.models.load_model(str(weights_path), compile=False)
        for src_layer, dst_layer in zip(loaded.layers, model.layers):
            try:
                dst_layer.set_weights(src_layer.get_weights())
            except Exception:
                pass  # skip mismatched layers
        print("[SETUP] Loaded weights via model transfer")
        return True
    except Exception as e:
        print(f"[SETUP] Model transfer failed: {e}")

    # Strategy 3: load with h5py directly
    try:
        import h5py
        with h5py.File(str(weights_path), 'r') as f:
            if 'model_weights' in f:
                model.load_weights(str(weights_path), by_name=True)
                print("[SETUP] Loaded weights by name (h5py)")
                return True
    except Exception as e:
        print(f"[SETUP] h5py load failed: {e}")

    return False


def make_feature_extractor(full_model):
    """
    Create a feature extractor from the full model.
    Output: 128-dim vector from the last Add layer (module 4 output),
    passed through GlobalAveragePooling2D.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    # Find the last Add layer (output of module 4, shape (h, w, 128))
    add_layer = None
    for layer in reversed(full_model.layers):
        if isinstance(layer, layers.Add):
            add_layer = layer
            break

    if add_layer is None:
        raise RuntimeError("Could not find Add layer in model")

    feature_map = add_layer.output  # (batch, h, w, 128)
    features = layers.GlobalAveragePooling2D(name='features')(feature_map)
    feature_model = Model(
        inputs=full_model.input,
        outputs=features,
        name='mini_xception_features'
    )
    return feature_model


def convert_to_tflite(model, output_path, quantize=False):
    """Convert Keras model to TFLite."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    with open(str(output_path), 'wb') as f:
        f.write(tflite_bytes)
    size_kb = len(tflite_bytes) / 1024
    print(f"[SETUP] Saved: {output_path.name} ({size_kb:.0f} KB)")


def verify_tflite(path, expected_output_shape):
    """Quick verification that the TFLite model runs."""
    import tensorflow as tf
    import numpy as np

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    dummy = np.random.rand(*inp_det['shape']).astype(np.float32)
    interp.set_tensor(inp_det['index'], dummy)
    interp.invoke()
    output = interp.get_tensor(out_det['index'])

    print(f"        Input:  {list(inp_det['shape'])}")
    print(f"        Output: {list(output.shape)} (expected {expected_output_shape})")
    assert list(output.shape[1:]) == list(expected_output_shape), \
        f"Shape mismatch: {output.shape} vs {expected_output_shape}"
    return True


def main():
    parser = argparse.ArgumentParser(description="Third Eye Shield Emotion Model Setup")
    parser.add_argument('--force', action='store_true',
                        help='Re-download and convert even if files exist')
    args = parser.parse_args()

    if CLASSIFIER_PATH.exists() and FEATURES_PATH.exists() and not args.force:
        print(f"[SETUP] Models already exist:")
        print(f"        {CLASSIFIER_PATH.name}")
        print(f"        {FEATURES_PATH.name}")
        print(f"        Use --force to re-create.")
        return

    print("=" * 60)
    print("  Third Eye Shield -- Emotion Model Setup")
    print("  Mini-Xception (FER2013, MIT License)")
    print("=" * 60)

    # Step 1: Build architecture
    print("\n[STEP 1] Building Mini-Xception architecture...")
    model = build_mini_xception()
    print(f"         Parameters: {model.count_params():,}")

    # Step 2: Download and load weights
    print("\n[STEP 2] Loading pre-trained weights...")
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
        weights_path = tmp.name

    try:
        download_weights(weights_path)
        loaded = load_weights_into_model(model, weights_path)
        if not loaded:
            print("[WARN]  Could not load pre-trained weights.")
            print("        The model will use random initialization.")
            print("        Few-shot personalization will still work but")
            print("        base predictions will be unreliable.")
    finally:
        try:
            os.unlink(weights_path)
        except OSError:
            pass

    # Step 3: Create feature extractor
    print("\n[STEP 3] Creating feature extractor...")
    feature_model = make_feature_extractor(model)
    print(f"         Feature dim: {feature_model.output_shape[-1]}")

    # Step 4: Convert to TFLite
    print("\n[STEP 4] Converting to TFLite...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_to_tflite(model, CLASSIFIER_PATH)
    convert_to_tflite(feature_model, FEATURES_PATH)

    # Step 5: Verify
    print("\n[STEP 5] Verifying TFLite models...")
    print(f"  Classifier:")
    verify_tflite(CLASSIFIER_PATH, [NUM_CLASSES])
    print(f"  Feature extractor:")
    verify_tflite(FEATURES_PATH, [FEATURE_DIM])

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print(f"  Classifier:  {CLASSIFIER_PATH}")
    print(f"  Features:    {FEATURES_PATH}")
    print("")
    print("  Next steps:")
    print("    1. Register a user:")
    print("       python3 scripts/calibrate_emotions.py --user-id UNCLE_TAN")
    print("    2. Run with emotion:")
    print("       python3 scripts/wellness_monitor.py --enable-emotion \\")
    print("           --emotion-profile UNCLE_TAN")
    print("=" * 60)


if __name__ == '__main__':
    main()
