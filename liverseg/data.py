"""LiverHccSeg data loading with PATIENT-WISE splits (no slice leakage between train/val/test).

Manifest CSV (one row per patient). Columns: patient_id, mask, and any of pre, art, pv, del
(paths relative to --data_root or absolute). Build it with scripts/make_manifest.py and CHECK it by hand.
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocessing import pfwf


def read_manifest(path):
    with open(path, newline="") as f:
        return [{k.strip().lower(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def split_patients(records, n_folds=5, fold=0, seed=42):
    """Patient-level k-fold: test = fold, val = next fold, train = the rest."""
    if n_folds < 3:
        raise ValueError("n_folds must be >= 3")
    idx = np.random.default_rng(seed).permutation(len(records))
    folds = np.array_split(idx, n_folds)
    test, val = folds[fold % n_folds], folds[(fold + 1) % n_folds]
    train = np.concatenate([f for j, f in enumerate(folds) if j not in (fold % n_folds, (fold + 1) % n_folds)])
    pick = lambda ii: [records[i] for i in ii]
    return pick(train), pick(val), pick(test)


def _resolve(root, p):
    return p if os.path.isabs(p) else os.path.join(root, p)


def _normalise(vol):
    lo, hi = np.percentile(vol, [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def load_patient(rec, root, phases, img_size=256, tumor_labels=(2,), use_pfwf=True):
    """Returns dict(id, images (Z,C,S,S) float16, masks (Z,S,S) uint8, spacing (dz,dy,dx) in mm)."""
    import nibabel as nib

    chans, zooms, shape = [], None, None
    for ph in phases:
        if not rec.get(ph):
            raise ValueError(f"patient {rec.get('patient_id')}: no file for phase '{ph}'")
        img = nib.load(_resolve(root, rec[ph]))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        if shape is None:
            shape, zooms = arr.shape, img.header.get_zooms()[:3]
        elif arr.shape != shape:
            raise ValueError(f"patient {rec.get('patient_id')}: phase '{ph}' shape {arr.shape} != {shape}")
        chans.append(_normalise(arr))
    seg = np.asarray(nib.load(_resolve(root, rec["mask"])).dataobj)
    if seg.ndim == 4:
        seg = seg[..., 0]
    if seg.shape != shape:
        raise ValueError(f"patient {rec.get('patient_id')}: mask shape {seg.shape} != image shape {shape}")
    mask = np.isin(seg, list(tumor_labels))

    X, Y, _ = shape
    imgs = torch.from_numpy(np.transpose(np.stack(chans, 0), (3, 0, 1, 2)).copy())      # Z,C,X,Y
    imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bilinear", align_corners=False)
    msk = torch.from_numpy(np.transpose(mask, (2, 0, 1)).copy()).float().unsqueeze(1)  # Z,1,X,Y
    msk = (F.interpolate(msk, size=(img_size, img_size), mode="nearest")[:, 0] > 0.5).to(torch.uint8)
    if use_pfwf:
        arr = imgs.numpy()
        for z in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                arr[z, c] = pfwf(arr[z, c])
        imgs = torch.from_numpy(arr)
    spacing = (float(zooms[2]), float(zooms[0]) * X / img_size, float(zooms[1]) * Y / img_size)
    return {"id": rec.get("patient_id", "?"), "images": imgs.half(), "masks": msk, "spacing": spacing}


def load_patients(records, cfg, tag=""):
    out = []
    for r in records:
        try:
            out.append(load_patient(r, cfg["data_root"], cfg["phases"].split(","), cfg["img_size"],
                                    tuple(int(v) for v in str(cfg["tumor_labels"]).split(",")),
                                    not cfg.get("no_pfwf", False)))
        except Exception as e:  # keep going, but make the problem visible
            print(f"[warn] skipping {r.get('patient_id')} ({tag}): {e}")
    print(f"[data] {tag}: {len(out)} patients, {sum(len(p['masks']) for p in out)} slices")
    return out


def load_fold(cfg):
    records = read_manifest(cfg["manifest"])
    tr, va, te = split_patients(records, cfg["n_folds"], cfg["fold"], cfg["seed"])
    return (load_patients(tr, cfg, "train"), load_patients(va, cfg, "val"), load_patients(te, cfg, "test"))


class SliceDataset(Dataset):
    """All tumour slices + a random fraction (neg_ratio) of tumour-free slices."""

    def __init__(self, patients, neg_ratio=0.2, augment=False, seed=0):
        self.patients, self.augment = patients, augment
        rng = np.random.default_rng(seed)
        self.index = []
        for pi, p in enumerate(patients):
            has = p["masks"].reshape(len(p["masks"]), -1).any(1).numpy()
            pos, neg = np.where(has)[0], np.where(~has)[0]
            k = int(round(len(neg) * neg_ratio))
            neg = rng.choice(neg, k, replace=False) if k > 0 else np.array([], dtype=int)
            self.index += [(pi, int(z)) for z in np.concatenate([pos, neg])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pi, z = self.index[i]
        p = self.patients[pi]
        x = p["images"][z].float()
        y = p["masks"][z].float().unsqueeze(0)
        if self.augment:
            if torch.rand(1) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k = int(torch.randint(0, 4, (1,)))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
            x = x * (1 + 0.1 * torch.randn(1)) + 0.05 * torch.randn(1)
        return x, y
