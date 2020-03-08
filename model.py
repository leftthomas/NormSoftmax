import torch
from capsule_layer.functional import k_means_routing
from torch import nn
from torch.nn import functional as F

from resnet import resnet18, resnet50, seresnet50


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class Pooling(nn.Module):
    def __init__(self, pooling_mode='map'):
        super().__init__()
        assert pooling_mode in ['k_means', 'map', 'avp'], 'pooling_mode {} is not supported'.format(pooling_mode)
        self.pooling_mode = pooling_mode

    def forward(self, x):
        assert x.dim() == 4, 'the input tensor must be the shape of [B, C, H, W]'
        if self.pooling_mode == 'map':
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        elif self.pooling_mode == 'avp':
            return torch.flatten(F.adaptive_avg_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            b, c, h, w = x.size()
            assert c % self.splits == 0, 'the channel of input tensor must be divided by {}'.format(self.splits)
            x = x.view(b, self.splits, c // self.splits, h * w)
            x = x.permute(0, 1, 3, 2).contiguous()
            y, _ = k_means_routing(x, num_iterations=1, similarity='cosine')
            y = y.view(b, c)
            return y

    def extra_repr(self):
        return 'pooling_mode={}'.format(self.pooling_mode)


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes, pooling_mode='map'):
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
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), nn.Linear(feature_dim, num_classes, bias=False))

    def forward(self, x):
        global_feature = self.features(x)
        global_feature = self.pooling(global_feature)
        feature = self.refactor(global_feature)
        classes = self.fc(feature)
        return F.normalize(feature, dim=-1), classes
