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


class FeatureFuse(nn.Module):
    def __init__(self, lower_channels, upper_channels):
        super(FeatureFuse, self).__init__()
        self.lower_channels = lower_channels
        self.upper_channels = upper_channels
        self.lower_conv = nn.Conv2d(lower_channels, lower_channels // 2, kernel_size=1, bias=False)
        self.upper_conv = nn.Conv2d(upper_channels, lower_channels // 2, kernel_size=1, bias=True)
        self.atten = nn.Conv2d(lower_channels // 2, out_channels=1, kernel_size=1, bias=True)
        self.W = nn.Sequential(nn.Conv2d(in_channels=lower_channels, out_channels=lower_channels, kernel_size=1),
                               nn.BatchNorm2d(lower_channels))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_lower, x_upper):
        b, c, h, w = x_lower.size()
        lower_feature = self.lower_conv(x_lower)
        upper_feature = F.interpolate(self.upper_conv(x_upper), size=(h, w), mode='bilinear', align_corners=True)
        feature = F.relu(lower_feature + upper_feature, inplace=True)
        atten = torch.sigmoid(self.atten(feature))
        return self.W(atten * x_lower)

    def extra_repr(self):
        return 'lower_channels={}, upper_channels={}'.format(self.lower_channels, self.upper_channels)


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

        # Feature Fuse
        self.fuse_2 = FeatureFuse(64 * expansion, 512 * expansion)
        self.fuse_3 = FeatureFuse(128 * expansion, 512 * expansion)

        # Refactor Layer
        self.refactor = nn.ModuleList([nn.Linear(64 * expansion, feature_dim // 12, bias=False),
                                       nn.Linear(128 * expansion, feature_dim // 6, bias=False),
                                       nn.Linear(512 * expansion, 3 * feature_dim // 4, bias=False)])
        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), ProxyLinear(feature_dim, num_classes))

    def forward(self, x):
        res0 = self.layer0(x)
        res1 = self.layer1(res0)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)
        fused_feature_2 = self.fuse_2(res1, res4)
        fused_feature_3 = self.fuse_3(res2, res4)
        fused_feature_2 = torch.flatten(F.adaptive_max_pool2d(fused_feature_2, output_size=(1, 1)), start_dim=1)
        fused_feature_3 = torch.flatten(F.adaptive_max_pool2d(fused_feature_3, output_size=(1, 1)), start_dim=1)
        global_feature = torch.flatten(F.adaptive_max_pool2d(res4, output_size=(1, 1)), start_dim=1)
        fused_feature_2 = self.refactor[0](fused_feature_2)
        fused_feature_3 = self.refactor[1](fused_feature_3)
        global_feature = self.refactor[2](global_feature)
        feature = torch.cat((fused_feature_2, fused_feature_3, global_feature), dim=-1)
        classes = self.fc(feature)
        return F.normalize(feature, dim=-1), classes
