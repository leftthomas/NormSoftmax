import math

import torch
from torch import nn
from torch.nn import functional as F

from resnet import resnet18, resnet50, seresnet50


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class Pooling(nn.Module):
    def __init__(self, pooling_mode='sap'):
        super().__init__()
        assert pooling_mode in ['sap', 'max', 'avg', 'gem'], 'pooling_mode {} is not supported'.format(pooling_mode)
        self.pooling_mode = pooling_mode

    def forward(self, x):
        assert x.dim() == 4, 'the input tensor must be the shape of [B, C, H, W]'
        if self.pooling_mode == 'max':
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        elif self.pooling_mode == 'avg':
            return torch.flatten(F.adaptive_avg_pool2d(x, output_size=(1, 1)), start_dim=1)
        elif self.pooling_mode == 'gem':
            sum_value = x.pow(self.p).mean(dim=[-1, -2])
            return torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / self.p))
        else:
            # [B, C, H*W]
            y = torch.flatten(x, start_dim=2)
            # [B, C]
            channel_attention = torch.bmm(y, y.permute(0, 2, 1).contiguous()).mean(dim=-1)
            channel_attention = channel_attention / channel_attention.max(dim=-1, keepdim=True)[0]
            # [B, H*W]
            spatial_attention = torch.bmm(y.permute(0, 2, 1).contiguous(), y).mean(dim=-1)
            spatial_attention = spatial_attention / spatial_attention.max(dim=-1, keepdim=True)[0]
            # [B, C, H*W]
            y = channel_attention.unsqueeze(dim=-1) * spatial_attention.unsqueeze(dim=1) * y
            # [B, C]
            y = torch.flatten(y, start_dim=1).topk(k=y.size(1), dim=-1, largest=True, sorted=False)[0]
            return y

    def extra_repr(self):
        return 'pooling_mode={}'.format(self.pooling_mode)


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
    def __init__(self, backbone_type, feature_dim, num_classes, pooling_mode='max'):
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
        self.pooling = Pooling(pooling_mode)
        self.refactor = nn.Linear(512 * expansion, feature_dim, bias=False)
        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), ProxyLinear(feature_dim, num_classes))

    def forward(self, x):
        global_feature = self.features(x)
        global_feature = self.pooling(global_feature)
        feature = self.refactor(global_feature)
        classes = self.fc(feature)
        return F.normalize(feature, dim=-1), classes
