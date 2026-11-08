"""Tests that need only numpy/scipy:  python tests/test_numpy.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from liverseg.preprocessing import pfwf, estimate_noise_sigma
from liverseg.metrics import segmentation_metrics

rng = np.random.default_rng(0)

# PFWF: output in range, same shape, reduces noise on a noisy gradient image
clean = np.tile(np.linspace(0.2, 0.8, 128), (128, 1)).astype(np.float32)
noisy = np.clip(clean + rng.normal(0, 0.08, clean.shape), 0, 1).astype(np.float32)
den = pfwf(noisy)
assert den.shape == noisy.shape and den.min() >= 0 and den.max() <= 1
assert np.abs(den - clean).mean() < np.abs(noisy - clean).mean(), "PFWF did not reduce noise"
assert estimate_noise_sigma(noisy * 255) > estimate_noise_sigma(clean * 255)

# metrics sanity
gt = np.zeros((20, 64, 64), bool); gt[5:15, 20:40, 20:40] = True
m = segmentation_metrics(gt, gt, spacing=(1, 1, 1))
assert m["dice"] == 1 and m["iou"] == 1 and m["voe"] == 0 and m["hd95"] == 0 and m["assd"] == 0
shift = np.roll(gt, 3, axis=2)
m = segmentation_metrics(shift, gt, spacing=(1, 1, 1))
assert 0.5 < m["dice"] < 1 and m["hd95"] >= 2.5 and m["assd"] > 0
m = segmentation_metrics(np.zeros_like(gt), gt, spacing=(1, 1, 1))
assert m["dice"] == 0 and np.isnan(m["hd95"])
print("numpy tests passed")
