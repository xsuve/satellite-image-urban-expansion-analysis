import torch.nn as nn
from models.UNet.UNetParts import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    def __init__(self, in_c, num_classes):
        super().__init__()

        # Encoder
        self.e1 = DownSample(in_c, 64)
        self.e2 = DownSample(64, 128)
        self.e3 = DownSample(128, 256)
        self.e4 = DownSample(256, 512)

        # Bottleneck
        self.b = DoubleConv(512, 1024)

        # Decoder
        self.d1 = UpSample(1024, 512)
        self.d2 = UpSample(512, 256)
        self.d3 = UpSample(256, 128)
        self.d4 = UpSample(128, 64)

        # Output
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck
        b = self.b(p4)

        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Output
        out = self.out(d4)
        return out
