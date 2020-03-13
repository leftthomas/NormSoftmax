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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.bn(x))


class PPMModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super().__init__()

        inter_channels = in_channels // len(sizes)
        assert in_channels % len(sizes) == 0

        self.stages = nn.ModuleList([self._make_stage(in_channels, inter_channels, size) for size in sizes])
        self.conv = ConvBlock(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def _make_stage(self, in_channels, inter_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = ConvBlock(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear',
                                align_corners=True) for stage in self.stages] + [feats]
        bottle = self.conv(torch.cat(priors, dim=1))
        return bottle


class FeatureFusion(nn.Module):
    def __init__(self, lower_channel, upper_channel):
        super().__init__()

        self.dwconv = ConvBlock(in_channels=lower_channel, out_channels=lower_channel, stride=1, padding=4, dilation=4,
                                groups=lower_channel)
        self.conv_high_res = nn.Conv2d(lower_channel, upper_channel, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv_low_res = nn.Conv2d(upper_channel, upper_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_res_input, low_res_input):
        b, c, h, w = low_res_input.size()
        high_res_input = F.interpolate(input=high_res_input, size=(h, w), mode='bilinear', align_corners=True)
        high_res_input = self.dwconv(high_res_input)
        high_res_input = self.conv_high_res(high_res_input)

        low_res_input = self.conv_low_res(low_res_input)

        x = torch.add(high_res_input, low_res_input)
        return self.relu(x)


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

        # PPM
        self.ppm = PPMModule(512 * expansion, 512 * expansion)
        self.fuse = FeatureFusion(64, 512 * expansion)

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
        global_feature = self.ppm(res4)
        global_feature = self.fuse(res0, global_feature)
        global_feature = torch.flatten(F.adaptive_max_pool2d(global_feature, output_size=(1, 1)), start_dim=1)
        feature = self.refactor(global_feature)
        classes = self.fc(feature)
        return F.normalize(feature, dim=-1), classes
