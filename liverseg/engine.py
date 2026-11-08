import random

import numpy as np
import torch

from .metrics import segmentation_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_volume(model, images, device, batch_size=16, tta=False):
    """images: (Z,C,S,S) -> probabilities (Z,S,S) on CPU."""
    model.eval()
    outs = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device).float()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)
                p = p + torch.sigmoid(model(x.flip(-2)).float()).flip(-2)
                p = p / 3
        outs.append(p[:, 0].cpu())
    return torch.cat(outs)


def evaluate_patients(model, patients, device, thr=0.5, surface=True, batch_size=16, tta=False):
    """3D per-patient metrics (DPC = mean of per-patient Dice) + pooled global Dice (DG)."""
    rows, inter, ps, gs = [], 0, 0, 0
    for p in patients:
        pred = predict_volume(model, p["images"], device, batch_size, tta).numpy() > thr
        gt = p["masks"].numpy().astype(bool)
        m = segmentation_metrics(pred, gt, spacing=p["spacing"], surface=surface)
        m["patient_id"] = p["id"]
        rows.append(m)
        inter += int((pred & gt).sum()); ps += int(pred.sum()); gs += int(gt.sum())
    dg = 2 * inter / (ps + gs) if (ps + gs) > 0 else 1.0
    return rows, dg


def summarize(rows, dg=None):
    keys = [k for k in rows[0] if k != "patient_id"] if rows else []
    s = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))}
         for k in keys}
    if dg is not None:
        s["dice_global"] = float(dg)
    return s
