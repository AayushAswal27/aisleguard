"""
BRIN — Bird's-eye Risk Inference Network.

A custom residual CNN, trained from scratch, that reads a 6-channel bird's-eye
velocity raster and forecasts forklift-pedestrian conflict risk as one of three
classes: SAFE / CAUTION / IMMINENT. Trained from scratch because the input
modality (signed velocity rasters) has no pretrained weights to transfer from.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A pre-activation-style residual block: two 3x3 convs with a skip connection."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)      # residual: learn the change, not the whole map


class BRIN(nn.Module):
    """
    BEV Risk Inference Network.

    Input:  (B, 6, 64, 64) bird's-eye velocity tensors
    Output: (B, 3) logits for SAFE / CAUTION / IMMINENT

    Architecture: stem -> res -> downsample -> res -> downsample -> res ->
    global average pool -> classifier head. Downsampling halves spatial size
    and doubles channels so that, by the final block, each cell's receptive
    field covers enough floor for two distant agents to interact.
    """

    def __init__(self, in_channels=6, num_classes=3):
        super().__init__()
        # stem: 6 channels -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU())
        self.res1  = ResidualBlock(32)                    # 64x64
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())                # 32x32
        self.res2  = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU())               # 16x16
        # final residual block: wide receptive field so distant agents interact
        self.res3  = ResidualBlock(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x); x = self.res2(x)
        x = self.down2(x); x = self.res3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)