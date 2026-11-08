"""Segmentation metrics in numpy (2D or 3D boolean arrays). Values are fractions (VOE = 1 - IoU)."""
import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    st = ndimage.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=st, border_value=0)


def surface_distances(pred, gt, spacing=None):
    """Returns (distances pred-surface->gt-surface, distances gt-surface->pred-surface)."""
    ps, gs = _surface(pred), _surface(gt)
    d_to_gt = ndimage.distance_transform_edt(~gs, sampling=spacing)
    d_to_pred = ndimage.distance_transform_edt(~ps, sampling=spacing)
    return d_to_gt[ps], d_to_pred[gs]


def hd95_assd(pred, gt, spacing=None):
    if not pred.any() or not gt.any():
        return float("nan"), float("nan")
    a, b = surface_distances(pred, gt, spacing)
    hd95 = max(np.percentile(a, 95), np.percentile(b, 95))
    assd = (a.sum() + b.sum()) / (len(a) + len(b))
    return float(hd95), float(assd)


def segmentation_metrics(pred, gt, spacing=None, surface=True) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    denom = 2 * tp + fp + fn
    dice = 1.0 if denom == 0 else 2 * tp / denom
    union = tp + fp + fn
    iou = 1.0 if union == 0 else tp / union
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    out = {"dice": float(dice), "iou": float(iou), "voe": float(1 - iou), "specificity": float(spec)}
    if surface:
        out["hd95"], out["assd"] = hd95_assd(pred, gt, spacing)
    return out
