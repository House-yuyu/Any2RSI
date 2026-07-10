"""
src/data/controls.py

Spatial control generation following ControlNet: Canny / HED / segmentation.
Canny is dependency-free via OpenCV. HED and Seg use AnyControl's pretrained
annotators when passed in; otherwise fall back to proxies so the pipeline runs.

For a faithful reproduction, wire the real annotators from AnyControl's
annotator/ directory (HED network, entity-seg / mask2former) into the dataset's
hed_annotator / seg_annotator args.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def canny(img, low=100, high=200):
    if not _HAS_CV2:
        g = img.mean(2)
        gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
        gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
        e = ((gx + gy) > 30).astype(np.uint8) * 255
        return np.repeat(e[:, :, None], 3, 2)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(gray, low, high)
    return np.repeat(e[:, :, None], 3, 2)


def hed(img, annotator=None):
    if annotator is not None:
        out = annotator(img)
        return out if out.ndim == 3 else np.repeat(out[:, :, None], 3, 2)
    return canny(img, 50, 150)


def segmentation(img, annotator=None):
    if annotator is not None:
        return annotator(img)
    if not _HAS_CV2:
        return ((img // 64) * 64).astype(np.uint8)
    Z = img.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, 6, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centers[labels.flatten()].reshape(img.shape).astype(np.uint8)


CONTROL_FNS = {"canny": canny, "hed": hed, "seg": segmentation}
