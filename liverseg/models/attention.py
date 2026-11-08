import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    """Pre-LN transformer layer (paper Eq. 4-5): x' = MSA(LN(x)) + x ; x = MLP(LN(x')) + x'."""

    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hid, dim), nn.Dropout(drop))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.msa(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TSA(nn.Module):
    """Transformer Self-Attention branch: patch embedding (p=1) + learnable position embedding
    (Eq. 3) + N transformer layers, reshaped back to a feature map."""

    def __init__(self, channels, dim, depth, heads, num_tokens, drop=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed = nn.Linear(channels, dim)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([TransformerLayer(dim, heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        if h * w != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} tokens, got {h * w}; img_size must match the model")
        t = self.embed(x.flatten(2).transpose(1, 2)) + self.pos
        for layer in self.layers:
            t = layer(t)
        t = self.out(self.norm(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class GSA(nn.Module):
    """Global Spatial Attention (Eq. 10-11): every position attends to every other position
    (query/key 1x1 convs -> softmax affinity -> weighted sum of value map)."""

    def __init__(self, channels, reduction=2):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.q = nn.Conv2d(channels, mid, 1)
        self.k = nn.Conv2d(channels, mid, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = mid ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.q(x).flatten(2), self.k(x).flatten(2), self.v(x).flatten(2)
        attn = torch.softmax(torch.einsum("bmi,bmj->bij", q, k) * self.scale, dim=-1)
        out = torch.einsum("bcj,bij->bci", v, attn)
        return out.reshape(b, c, h, w)


class SelfAwareAttention(nn.Module):
    """Fusion of TSA and GSA (Eq. 12): out = x + lam1*TSA(x) + lam2*GSA(x).
    lam1, lam2 are learnable scalars initialised to `lambda_init` (paper: 0, then grown by training).
    The identity path keeps the block safe when lambdas are still ~0."""

    def __init__(self, channels, dim, depth, heads, num_tokens, use_tsa=True, use_gsa=True, lambda_init=0.0):
        super().__init__()
        self.tsa = TSA(channels, dim, depth, heads, num_tokens) if use_tsa else None
        self.gsa = GSA(channels) if use_gsa else None
        self.lam1 = nn.Parameter(torch.full((1,), float(lambda_init)))
        self.lam2 = nn.Parameter(torch.full((1,), float(lambda_init)))

    def forward(self, x):
        out = x
        if self.tsa is not None:
            out = out + self.lam1 * self.tsa(x)
        if self.gsa is not None:
            out = out + self.lam2 * self.gsa(x)
        return out


class DirectionGuidance(nn.Module):
    """Direction-guidance block (Eq. 8): a small head predicts a 2-channel direction field DI, and
    the feature map is iteratively re-sampled f_t(p) = f_{t-1}(p - DI(p)) so features drift toward
    object boundaries. DI is learned end-to-end (the paper gives no explicit supervision)."""

    def __init__(self, channels, steps=2, max_shift=0.1):
        super().__init__()
        self.steps, self.max_shift = steps, max_shift
        self.head = nn.Sequential(conv_bn(channels, channels // 2), nn.Conv2d(channels // 2, 2, 1))
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, feat):
        b, _, h, w = feat.shape
        di = torch.tanh(self.head(feat)) * self.max_shift          # (b,2,h,w), normalised coords
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=feat.device),
                                torch.linspace(-1, 1, w, device=feat.device), indexing="ij")
        base = torch.stack((xs, ys), dim=-1).unsqueeze(0).to(di.dtype)   # (1,h,w,2)
        grid = base - di.permute(0, 2, 3, 1)
        out = feat
        for _ in range(self.steps):
            out = F.grid_sample(out, grid.to(out.dtype), mode="bilinear",
                                padding_mode="border", align_corners=True)
        return self.refine(out) + feat


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
