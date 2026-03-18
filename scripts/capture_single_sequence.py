#!/usr/bin/env python3
"""
Simple skeleton capture for few-shot data collection
Captures one sequence and saves it, then exits.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pyrealsense2 as rs
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.movenet_pose_extractor import MoveNetPoseExtractor


def capture_one_sequence(seq_len=48, conf_threshold=0.20, timeout=15):
    """Capture exactly one skeleton sequence and return it."""
    # Pose extractor (single-pose Lightning int8)
    extractor = MoveNetPoseExtractor(
        model_path='models/movenet_lightning_int8.tflite',
        conf_threshold=conf_threshold
    )
    
    # RealSense setup
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[ERROR] No RealSense device detected", file=sys.stderr)
        return None
    
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    
    # Warm-up
    for _ in range(5):
        pipeline.wait_for_frames()
    
    buffer = []
    start_time = time.time()
    
    try:
        while len(buffer) < seq_len:
            if time.time() - start_time > timeout:
                print(f"[WARN] Timeout after {timeout}s, only captured {len(buffer)} frames", file=sys.stderr)
                break
            
            frames = pipeline.wait_for_frames()
            d = frames.get_depth_frame()
            c = frames.get_color_frame()
            if not d or not c:
                continue
            
            depth = np.asanyarray(d.get_data())
            rgb = np.asanyarray(c.get_data())
            
            # Pose extraction (multipose: primary skeleton auto-selected)
            skeleton, _, detected = extractor.extract(rgb, draw=False)
            
            if detected and skeleton is not None:
                buffer.append(skeleton.copy())  # skeleton is (17, 3)
                if len(buffer) % 10 == 0:
                    print(f"[INFO] Captured {len(buffer)}/{seq_len} frames...", file=sys.stderr)
    
    finally:
        pipeline.stop()
    
    if len(buffer) >= seq_len:
        # Stack to (T, 17, 3)
        sequence = np.stack(buffer[:seq_len], axis=0)
        return sequence
    else:
        print(f"[ERROR] Failed to capture {seq_len} frames, only got {len(buffer)}", file=sys.stderr)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='Output .npz path')
    parser.add_argument('--seq', type=int, default=48, help='Sequence length')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold')
    parser.add_argument('--timeout', type=int, default=15, help='Capture timeout (seconds)')
    args = parser.parse_args()
    
    print(f"[INFO] Capturing {args.seq}-frame sequence...", file=sys.stderr)
    sequence = capture_one_sequence(args.seq, args.conf, args.timeout)
    
    if sequence is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, keypoints=sequence)
        print(f"[OK] Saved sequence to {out_path}", file=sys.stderr)
        return 0
    else:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
