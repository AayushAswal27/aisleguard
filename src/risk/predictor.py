import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
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
        return F.relu(out + identity)


class BRIN(nn.Module):
    """
    BEV Risk Interaction Network.
    Input:  (B, 6, 64, 64) bird's-eye tensors
    Output: (B, 3) logits for SAFE / CAUTION / IMMINENT
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
        # dilated block: wide receptive field so distant agents interact
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