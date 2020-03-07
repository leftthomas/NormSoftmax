import torch
from torch import nn
from torch.nn import functional as F

from resnet import resnet18, resnet50


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes):
        super().__init__()

        # Backbone Network
        backbones = {'resnet18': (resnet18, 1), 'resnet50': (resnet50, 4)}
        backbone, expansion = backbones[backbone_type]
        self.features = []
        for name, module in backbone(pretrained=True).named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Linear):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)

        # Refactor Layer
        self.refactor = nn.Linear(512 * expansion, feature_dim, bias=False)
        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), nn.Linear(feature_dim, num_classes, bias=False))

    def forward(self, x):
        global_feature = F.adaptive_max_pool2d(self.features(x), output_size=(1, 1))
        global_feature = torch.flatten(global_feature, start_dim=1)
        feature = self.refactor(global_feature)
        classes = self.fc(feature)
        return F.normalize(feature, dim=-1), classes
