"""TFLite depth autoencoder wrapper for anomaly detection."""
import numpy as np


class DepthAutoencoder:
    """Depth autoencoder TFLite wrapper.

    Computes reconstruction MSE on a depth frame as an anomaly score.
    Lower MSE = better reconstruction = person present = anomaly.
    """

    def __init__(self, model_path, input_size=(64, 64)):
        import tensorflow as tf
        self.size = input_size
        interp = tf.lite.Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        self._interp = interp
        self._inp_idx = interp.get_input_details()[0]['index']
        self._out_idx = interp.get_output_details()[0]['index']

    def mse(self, depth_raw):
        """Compute reconstruction MSE for a raw uint16 depth frame.

        Args:
            depth_raw: (H, W) uint16 depth image in millimetres.

        Returns:
            float: Mean squared error between input and reconstruction.
        """
        import cv2
        d = cv2.resize(depth_raw, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        d = np.clip((d - 100) / 9900.0, 0, 1)[None, ..., None]
        self._interp.set_tensor(self._inp_idx, d)
        self._interp.invoke()
        rec = self._interp.get_tensor(self._out_idx)[0, ..., 0]
        return float(np.mean((d[0, ..., 0] - rec) ** 2))
