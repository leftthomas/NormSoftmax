import math

import torch
import torch.nn.functional as F
from torch import nn

from resnet import resnet18, resnet50, seresnet50


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class ProxyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProxyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        output = x.matmul(self.weight.t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class WeightPooling(nn.Module):
    def __init__(self, channel, height, width, num_point):
        super(WeightPooling, self).__init__()
        self.height = height
        self.width = width
        self.num_point = num_point
        self.spatial_attention = nn.Parameter(torch.Tensor(1, channel, height, width))
        self.reduce = nn.Conv1d(channel, channel, num_point, groups=channel, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.spatial_attention, 1)

    def forward(self, x):
        # [B, C, H*W]
        output = torch.flatten(self.spatial_attention * x, start_dim=2)
        # [B, C, N]
        output, locations = output.topk(k=self.num_point, dim=-1)
        # [B, C]
        output = torch.flatten(self.reduce(output), start_dim=1)
        return output, locations.float()

    def extra_repr(self):
        return 'height={}, width={}, num_point={}'.format(self.height, self.width, self.num_point)


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes):
        super().__init__()

        # Backbone Network
        backbones = {'resnet18': (resnet18, 1), 'resnet50': (resnet50, 4), 'seresnet50': (seresnet50, 4)}
        backbone, expansion = backbones[backbone_type]
        self.layer0 = []
        for name, module in backbone(pretrained=True).named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Linear):
                continue
            if name not in ['layer1', 'layer2', 'layer3', 'layer4']:
                self.layer0.append(module)
            else:
                self.add_module(name, module)
        self.layer0 = nn.Sequential(*self.layer0)

        self.pooling = WeightPooling(512 * expansion, 15, 15, 4)

        # Refactor Layer
        self.refactor = nn.Linear(512 * expansion, feature_dim, bias=False)
        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), ProxyLinear(feature_dim, num_classes))

    def forward(self, x):
        res0 = self.layer0(x)
        res1 = self.layer1(res0)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)
        global_feature, locations = self.pooling(res4)
        feature = self.refactor(global_feature)
        classes = self.fc(feature)
        return F.normalize(feature, dim=-1), classes, locations
