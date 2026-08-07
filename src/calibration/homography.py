import numpy as np
import cv2
import yaml


def load_homography(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return np.array(cfg["homography"], dtype=np.float32)


def pixel_to_metres(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0][0]
    return float(out[0]), float(out[1])