import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze=16, expand=64):
        super(Fire, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, squeeze, 1)
        self.relu = nn.ReLU()
        self.conv2d_l = nn.Conv2d(squeeze, expand, 1)
        self.conv2d_r = nn.Conv2d(squeeze, expand, 3, padding=1)
        pass

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        left = self.conv2d_l(x)
        right = self.conv2d_r(x)

        return torch.cat([left, right], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, input_shape):
        super(SqueezeNet, self).__init__()
        h, w = input_shape
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        h = (((h - 3) // 2 + 1) - 3) // 2 + 1
        w = (((w - 3) // 2 + 1) - 3) // 2 + 1
        self.block_2 = nn.Sequential(
            Fire(64, 16, 16),
            Fire(32, 16, 16),
            nn.MaxPool2d(3, 2)
        )
        h = (h - 3) // 2 + 1
        w = (w - 3) // 2 + 1
        self.block_3 = nn.Sequential(
            Fire(32, 32, 32),
            Fire(64, 32, 32),
            nn.MaxPool2d(3, 2)
        )
        h = (h - 3) // 2 + 1
        w = (w - 3) // 2 + 1
        self.block_4 = nn.Sequential(
            Fire(64, 48, 48),
            Fire(96, 48, 48),
            Fire(96, 64, 64),
            Fire(128, 64, 64),
            nn.Dropout2d(0.5)
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(128, 5, 1),
            nn.ReLU(),
        )

        self.linear = nn.Linear(5*w*h, 1)
        pass

    def forward(self, x):
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
