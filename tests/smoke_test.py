"""Needs torch.  Run FIRST on your machine:  python tests/smoke_test.py
Checks every ablation variant builds, runs forward/backward and gives the right output shape."""
import os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from liverseg.models import MDASGT
from liverseg.losses import DiceBCELoss

S = 64  # small for speed; must be divisible by 16
variants = [
    dict(),
    dict(encoder="plain"),
    dict(use_transformer=False),
    dict(use_gsa=False),
    dict(use_transformer=False, use_gsa=False, direction_guidance=False),
    dict(skip="unet"),
    dict(direction_guidance=False, lambda_init=0.1),
    dict(in_ch=3),
]
for kw in variants:
    m = MDASGT(img_size=S, base=16, embed_dim=64, depth=2, heads=4, **kw)
    x = torch.randn(2, kw.get("in_ch", 1), S, S)
    y = (torch.rand(2, 1, S, S) > 0.8).float()
    out = m(x)
    assert out.shape == (2, 1, S, S), (kw, out.shape)
    loss = DiceBCELoss()(out, y)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()), kw
    print("ok", kw or "default", f"loss={loss.item():.3f}")
print("smoke test passed")
