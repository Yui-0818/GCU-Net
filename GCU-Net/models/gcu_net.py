import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, d=1):
        super().__init__()
        if p is None:
            p = d if k == 3 else 0
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = ConvBNReLU(cin, cout)
        self.c2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, skip_c, gate_c):
        super().__init__()
        mid = max(skip_c // 2, 8)
        self.theta = nn.Conv2d(skip_c, mid, 1, bias=False)
        self.phi = nn.Conv2d(gate_c, mid, 1, bias=False)
        self.psi = nn.Conv2d(mid, 1, 1)

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, x.shape[-2:], mode="bilinear", align_corners=False)
        a = torch.sigmoid(self.psi(F.relu(self.theta(x) + self.phi(g), inplace=True)))
        return x * a


class ASPP(nn.Module):
    def __init__(self, c, branch=64):
        super().__init__()
        self.b1 = ConvBNReLU(c, branch, 1)
        self.b2 = ConvBNReLU(c, branch, 3, d=3)
        self.b3 = ConvBNReLU(c, branch, 3, d=6)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, branch, 1, bias=False),
            nn.GroupNorm(8, branch),
            nn.ReLU(inplace=True),
        )
        self.fuse = ConvBNReLU(branch * 4, c, 1)

    def forward(self, x):
        size = x.shape[-2:]
        out = torch.cat([
            self.b1(x),
            self.b2(x),
            self.b3(x),
            F.interpolate(self.pool(x), size, mode="bilinear", align_corners=False),
        ], dim=1)
        return self.fuse(out)


class GCU_Net(nn.Module):
    """Green-guided corneal ulcer network: 3-channel input, residual encoder,
    attention-gated decoder, ASPP bottleneck and two deep-supervision heads."""

    def __init__(self, in_ch=3, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvBNReLU(in_ch, c1)
        self.e1 = ResBlock(c1, c1)
        self.e2 = nn.Sequential(ConvBNReLU(c1, c2, s=2), ResBlock(c2, c2), ResBlock(c2, c2))
        self.e3 = nn.Sequential(ConvBNReLU(c2, c3, s=2), ResBlock(c3, c3), ResBlock(c3, c3))
        self.e4 = nn.Sequential(ConvBNReLU(c3, c4, s=2), ResBlock(c4, c4), ResBlock(c4, c4))
        self.e5 = nn.Sequential(ConvBNReLU(c4, c4, s=2), ResBlock(c4, c4), ASPP(c4))

        self.ag4 = AttentionGate(c4, c4)
        self.ag3 = AttentionGate(c3, c3)
        self.ag2 = AttentionGate(c2, c2)
        self.ag1 = AttentionGate(c1, c1)

        self.fuse4 = ConvBNReLU(c4 + c4, c3, 3)
        self.fuse3 = ConvBNReLU(c3 + c3, c2, 3)
        self.fuse2 = ConvBNReLU(c2 + c2, c1, 3)
        self.fuse1 = ConvBNReLU(c1 + c1, c1 // 2, 3)
        self.d4 = ResBlock(c3, c3)
        self.d3 = ResBlock(c2, c2)
        self.d2 = ResBlock(c1, c1)
        self.d1 = ResBlock(c1 // 2, c1 // 2)

        self.head = nn.Conv2d(c1 // 2, 1, 3, 1, 1)
        self.aux_half = nn.Conv2d(c1, 1, 3, 1, 1)
        self.aux_quarter = nn.Conv2d(c2, 1, 3, 1, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            if m.out_channels == 1:
                nn.init.normal_(m.weight, std=0.01)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        s1 = self.e1(self.stem(x))
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b = self.e5(s4)

        d4 = self.d4(self.fuse4(torch.cat([F.interpolate(b, s4.shape[-2:], mode="bilinear",
                                                          align_corners=False),
                                           self.ag4(s4, b)], dim=1)))
        d3 = self.d3(self.fuse3(torch.cat([F.interpolate(d4, s3.shape[-2:], mode="bilinear",
                                                          align_corners=False),
                                           self.ag3(s3, d4)], dim=1)))
        d2 = self.d2(self.fuse2(torch.cat([F.interpolate(d3, s2.shape[-2:], mode="bilinear",
                                                          align_corners=False),
                                           self.ag2(s2, d3)], dim=1)))
        d1 = self.d1(self.fuse1(torch.cat([F.interpolate(d2, s1.shape[-2:], mode="bilinear",
                                                          align_corners=False),
                                           self.ag1(s1, d2)], dim=1)))
        main = self.head(d1)
        if self.training:
            return [main, self.aux_half(d2), self.aux_quarter(d3)]
        return main
