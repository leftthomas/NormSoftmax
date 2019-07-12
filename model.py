import torch
import torch.nn as nn
from capsule_layer import CapsuleLinear
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, meta_classes, ensemble_size):
        super(Model, self).__init__()

        # backbone
        self.meta_classes, self.ensemble_size = meta_classes, ensemble_size

        # common features
        basic_model, self.common_extractor = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if name == 'conv1' or name == 'bn1' or name == 'relu' or name == 'maxpool' or name == 'layer1':
                self.common_extractor.append(module)
            else:
                continue
        self.common_extractor = nn.Sequential(*self.common_extractor)

        # sole features
        self.sole_extractors = []
        for _ in range(ensemble_size):
            basic_model, layers = resnet18(pretrained=True), []
            for name, module in basic_model.named_children():
                if name == 'layer2' or name == 'layer3' or name == 'layer4':
                    layers.append(module)
                else:
                    continue
            layers = nn.Sequential(*layers)
            self.sole_extractors.append(layers)
        self.sole_extractors = nn.ModuleList(self.sole_extractors)

        # attention block
        self.sole_attentions = nn.ModuleList([CapsuleLinear(16, 512, 128) for _ in range(ensemble_size)])

        # sole classifiers
        self.classifiers = nn.ModuleList([nn.Sequential(nn.Linear(2048, meta_classes)) for _ in range(ensemble_size)])

    def forward(self, x):
        common_feature = self.common_extractor(x)
        out = []
        for i in range(self.ensemble_size):
            sole_feature = self.sole_extractors[i](common_feature)
            sole_feature = sole_feature.permute(0, 2, 3, 1).contiguous().view(sole_feature.size(0), -1, 128)
            att_feature = self.sole_attentions[i](sole_feature)
            att_feature = att_feature.view(att_feature.size(0), -1)
            sole_classes = self.classifiers[i](att_feature)
            out.append(sole_classes)
        out = torch.stack(out, dim=1)
        return out
