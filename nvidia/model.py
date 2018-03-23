import torch
import torch.nn as nn


class NvidiaNet(nn.Module):
    def __init__(self, input_shape):
        super(NvidiaNet, self).__init__()
        h, w = input_shape
        self.bn = nn.BatchNorm2d(3)
        self.conv_1 = nn.Conv2d(3, 24, 5, 2)
        h = (h - 5) // 2 + 1
        w = (w - 5) // 2 + 1
        self.conv_2 = nn.Conv2d(24, 36, 5, 2)
        h = (h - 5) // 2 + 1
        w = (w - 5) // 2 + 1
        self.conv_3 = nn.Conv2d(36, 48, 5, 2)
        h = (h - 5) // 2 + 1
        w = (w - 5) // 2 + 1
        self.conv_4 = nn.Conv2d(48, 64, 3, 1)
        h = (h - 3) + 1
        w = (w - 3) + 1
        self.conv_5 = nn.Conv2d(64, 64, 3, 1)
        h = (h - 3) + 1
        w = (w - 3) + 1

        self.linear_1 = nn.Linear(64 * h * w, 100)
        self.linear_2 = nn.Linear(100, 50)
        self.linear_3 = nn.Linear(50, 10)
        self.linear_4 = nn.Linear(10, 1)
        pass

    def forward(self, x):
        out = self.bn(x)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = out.view(out.size(0), -1)
        out = self.linear_1(out)
        out = self.linear_2(out)
        out = self.linear_3(out)
        out = self.linear_4(out)

        return out
