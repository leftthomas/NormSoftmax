import torch
from torch import nn
from torch.nn import functional as F

from resnet import resnet18, resnet50, seresnet50


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim, reduction=16):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # [B, C/16, 7, 7]
            nn.Conv2d(in_channels, in_channels // reduction, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(True),
            # [B, C/16, 4, 4]
            nn.MaxPool2d(3, stride=1, padding=1),
            # [B, D, 2, 2]
            # nn.Conv2d(in_channels // reduction, feature_dim, 3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(feature_dim),
            # nn.ReLU(True),
            # [B, D, 1, 1]
            # nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, in_channels // reduction, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels // reduction, in_channels // reduction, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels // reduction, in_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes):
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
        self.refactor = AutoEncoder(512 * expansion, feature_dim)
        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), nn.Linear(feature_dim, num_classes, bias=False))

    def forward(self, x):
        global_feature = self.features(x)
        encoder, decoder = self.refactor(global_feature)
        feature = torch.flatten(encoder, start_dim=1)
        classes = self.fc(feature)
        return F.normalize(feature, dim=-1), classes, decoder, global_feature
