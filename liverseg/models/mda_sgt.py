import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, Encoder
from .attention import SelfAwareAttention, DirectionGuidance


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class MDASGT(nn.Module):
    """CRes2Net encoder + Self-aware attention (TSA+GSA) bottleneck + multi-scale-skip decoder.

    Switches (used for the ablation study):
      encoder            'cres2net' | 'plain'
      use_transformer    TSA branch on/off
      use_gsa            GSA branch on/off  (attention bottleneck is skipped if both are off)
      skip               'dense' (paper Eq. 15: each decoder level sees ALL deeper features) | 'unet'
      direction_guidance direction-guidance block on the last decoder feature map
    img_size must be divisible by 16 and equal to the training image size.
    """

    def __init__(self, in_ch=1, num_classes=1, base=32, img_size=256, encoder="cres2net",
                 use_transformer=True, use_gsa=True, skip="dense", direction_guidance=True,
                 embed_dim=384, depth=4, heads=8, scale=4, lambda_init=0.0):
        super().__init__()
        assert img_size % 16 == 0
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.dense = skip == "dense"
        self.encoder = Encoder(in_ch, chs, kind=encoder, scale=scale)

        self.attn = None
        if use_transformer or use_gsa:
            tokens = (img_size // 16) ** 2
            self.attn = SelfAwareAttention(chs[4], embed_dim, depth, heads, tokens,
                                           use_tsa=use_transformer, use_gsa=use_gsa, lambda_init=lambda_init)

        # decoder levels from /8 up to /1; level i consumes enc[i] + (all | last) deeper features
        self.dec = nn.ModuleList()
        deep_ch = [chs[4]]
        for i in range(3, -1, -1):
            deeper = sum(deep_ch) if self.dense else deep_ch[-1]
            self.dec.append(ConvBlock(chs[i] + deeper, chs[i]))
            deep_ch.append(chs[i])
        self.dg = DirectionGuidance(chs[0]) if direction_guidance else None
        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        feats = self.encoder(x)
        enc, b = feats[:4], feats[4]
        if self.attn is not None:
            b = self.attn(b)
        deep = [b]
        for idx, i in enumerate(range(3, -1, -1)):
            srcs = deep if self.dense else deep[-1:]
            d = self.dec[idx](torch.cat([enc[i]] + [_up(s, enc[i]) for s in srcs], dim=1))
            deep.append(d)
        d = deep[-1]
        if self.dg is not None:
            d = self.dg(d)
        return self.head(d)   # logits
