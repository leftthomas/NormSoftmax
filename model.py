import math

import torch
import torch.nn.functional as F
from torch import nn

from resnet import resnet18, resnet50, seresnet50


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class Refactor(nn.Module):
    def __init__(self, in_features, out_features, refactor_mode='tra'):
        super(Refactor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.refactor_mode = refactor_mode
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        if self.refactor_mode == 'tra':
            x = torch.flatten(F.adaptive_avg_pool2d(x, output_size=(1, 1)), start_dim=1)
        elif self.refactor_mode == 'max':
            x = torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        elif self.refactor_mode == 'avg':
            x = torch.flatten(F.adaptive_avg_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            sum_value = x.pow(3).mean(dim=[-1, -2])
            x = torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / 3))
        output = self.fc(x)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


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


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes, refactor_mode='tra'):
        super().__init__()

        # Backbone Network
        backbones = {'resnet18': (resnet18, 1), 'resnet50': (resnet50, 4), 'seresnet50': (seresnet50, 4)}
        backbone, expansion = backbones[backbone_type]
        self.features = []
        for name, module in backbone(pretrained=True).named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Linear):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)

        # Refactor Layer
        self.refactor = Refactor(512 * expansion, feature_dim, refactor_mode)
        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), ProxyLinear(feature_dim, num_classes))

    def forward(self, x):
        global_feature = self.features(x)
        feature = self.refactor(global_feature)
        classes = self.fc(feature)
        return F.normalize(feature, dim=-1), classes
