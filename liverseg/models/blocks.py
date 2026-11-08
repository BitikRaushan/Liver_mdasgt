import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ConvBlock(nn.Module):
    """Plain double 3x3 conv (decoder blocks, and the 'plain' encoder used for ablation)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: GAP -> FC -> ReLU -> FC -> sigmoid -> channel re-scaling."""

    def __init__(self, c, reduction=16):
        super().__init__()
        m = max(c // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(c, m), nn.ReLU(inplace=True), nn.Linear(m, c), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.fc(x.mean((2, 3))).view(b, c, 1, 1)
        return x * w


class Res2NetSEBlock(nn.Module):
    """Res2Net block (paper Eq. 2) followed by an SE block, with a residual shortcut.

    1x1 conv -> split into `scale` groups G1..Gs -> M1=G1, M2=conv(G2), Mi=conv(Gi + M(i-1))
    -> concat -> 1x1 conv -> SE -> + shortcut.
    """

    def __init__(self, cin, cout, scale=4, reduction=16):
        super().__init__()
        assert cout % scale == 0, "out channels must be divisible by scale"
        self.scale = scale
        w = cout // scale
        self.conv1 = conv_bn_relu(cin, cout, k=1)
        self.convs = nn.ModuleList([conv_bn_relu(w, w, k=3) for _ in range(scale - 1)])
        self.conv3 = nn.Sequential(nn.Conv2d(cout, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.se = SEBlock(cout, reduction)
        self.short = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        chunks = torch.chunk(self.conv1(x), self.scale, dim=1)
        outs, prev = [chunks[0]], None
        for i in range(1, self.scale):
            inp = chunks[i] if prev is None else chunks[i] + prev
            prev = self.convs[i - 1](inp)
            outs.append(prev)
        y = self.se(self.conv3(torch.cat(outs, dim=1)))
        return self.act(y + self.short(x))


class Encoder(nn.Module):
    """5-level encoder. kind='cres2net' uses Res2Net+SE blocks, kind='plain' uses double convs.
    Returns a list of 5 feature maps (resolution /1, /2, /4, /8, /16)."""

    def __init__(self, in_ch, chs, kind="cres2net", scale=4):
        super().__init__()
        self.stem = conv_bn_relu(in_ch, chs[0])
        self.pool = nn.MaxPool2d(2)
        stages = []
        for i, c in enumerate(chs):
            cin = chs[0] if i == 0 else chs[i - 1]
            if kind == "cres2net":
                stages.append(nn.Sequential(Res2NetSEBlock(cin, c, scale), Res2NetSEBlock(c, c, scale)))
            elif kind == "plain":
                stages.append(ConvBlock(cin, c))
            else:
                raise ValueError(f"unknown encoder kind: {kind}")
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for i, st in enumerate(self.stages):
            if i > 0:
                x = self.pool(x)
            x = st(x)
            feats.append(x)
        return feats
