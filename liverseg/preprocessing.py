"""Probabilistic Fused Wiener Filter (PFWF) - numpy/scipy only.

The paper's Table 2 is ambiguous, so this is an *interpretation*:
  1. Noise sigma  : estimated by convolving with a Laplacian-type mask and averaging |response|
                    (Immerkaer's fast estimator - same form as the paper's Step 2 formula).
  2. Mask size    : 3x3 if sigma < 20 else 5x5 (Step 3, image on a 0-255 scale).
  3. Mean filter  : local mean (Step 4).
  4. Wiener filter: scipy.signal.wiener with the same window.
  5. Variation    : Sl = |G - mean_filtered| (Step 5).
  6. 'Probabilistic' fusion (Steps 6-7): where the deviation is below its mean (smooth region)
                    output the average of the mean-filtered and Wiener results (uniform weights);
                    elsewhere (edges / tumour boundaries) keep the Wiener result to preserve detail.
"""
import numpy as np
from scipy import ndimage, signal

_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimate_noise_sigma(img255: np.ndarray) -> float:
    h, w = img255.shape
    conv = ndimage.convolve(img255.astype(np.float64), _MASK, mode="reflect")[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv).sum() / (6.0 * (h - 2) * (w - 2)))


def pfwf(img: np.ndarray, sigma_threshold: float = 20.0) -> np.ndarray:
    """Denoise a 2D image with values in [0, 1]; returns float32 in [0, 1]."""
    g = np.clip(img, 0.0, 1.0).astype(np.float64) * 255.0
    sigma = estimate_noise_sigma(g)
    k = 3 if sigma < sigma_threshold else 5
    mean_f = ndimage.uniform_filter(g, size=k, mode="reflect")
    wien = signal.wiener(g, mysize=k)
    wien = np.where(np.isfinite(wien), wien, g)
    sl = np.abs(g - mean_f)
    smooth = sl < sl.mean()
    out = np.where(smooth, 0.5 * (mean_f + wien), wien)
    return np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)
