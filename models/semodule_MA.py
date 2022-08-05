from torch import nn
import torch
import torch.nn.functional as F
import pdb


def tensor_norm(a):
    a_max, _ = a.max(1)
    a_min, _ = a.min(1)
    a_max = a_max.reshape(-1,1).expand_as(a)
    a_min = a_min.reshape(-1,1).expand_as(a)

    a = (a-a_min)/(a_max-a_min)
    return a


def feature_norm(a):
    return (a - a.min())/(a.max() - a.min())


class GCBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GCBAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.ReLU(inplace=True),

        )

        self.fc1 = nn.Linear(channel//3, 1, bias=False)

    def forward(self, x):
        layer = len(x)
        b, c, _, _ = x[0].size()
        x0 = feature_norm(x[0])
        x1 = feature_norm(x[1])
        x2 = feature_norm(x[2])

        ay0 = self.avg_pool(x0).view(b, c)

        ay1 = self.avg_pool(x1).view(b, c)

        ay2 = self.avg_pool(x2).view(b, c)

        ay = torch.cat([ay0, ay1, ay2], 1)

        ay = self.fc(ay)

        y = (ay).view(b * layer, -1)
        y = self.fc1(y)
        y = F.sigmoid(y).view(b, -1,1,1,1)

        return [y[:, i,:,:,:].expand_as(x[i]) for i in range(layer)]


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        ay = self.avg_pool(x).view(b, c)
        my = self.max_pool(x).view(b, c)

        ay = self.fc(ay)
        my = self.fc(my)

        y = F.sigmoid(ay + my).view(b, c, 1, 1)
        return y.expand_as(x)
