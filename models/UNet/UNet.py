import torch.nn as nn
from models.UNet.UNetParts import UNetConv, UNetEncoder, UNetDecoder


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # Encoder
        self.e1 = UNetEncoder(3, 64)
        self.e2 = UNetEncoder(64, 128)
        self.e3 = UNetEncoder(128, 256)
        self.e4 = UNetEncoder(256, 512)

        # Bottleneck
        self.b = UNetConv(512, 1024)

        # Decoder
        self.d1 = UNetDecoder(1024, 512)
        self.d2 = UNetDecoder(512, 256)
        self.d3 = UNetDecoder(256, 128)
        self.d4 = UNetDecoder(128, 64)

        # Output
        self.outputs = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)
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

        # output
        outputs = self.outputs(d4)
        return outputs
