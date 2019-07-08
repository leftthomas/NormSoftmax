import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, meta_class_size, ensemble_size):
        super(Model, self).__init__()

        # backbone
        self.meta_class_size, self.ensemble_size = meta_class_size, ensemble_size
        basic_model = resnet18(pretrained=True)

        # common features
        self.common_extractor = []
        for name, module in basic_model.named_children():
            if name == 'conv1' or name == 'bn1' or name == 'relu' or name == 'maxpool':
                self.common_extractor.append(module)
            else:
                continue
        self.common_extractor = nn.Sequential(*self.common_extractor)

        # individual features
        self.individual_extractors = []
        for i in range(ensemble_size):
            layers = []
            for name, module in basic_model.named_children():
                if name == 'layer1' or name == 'layer2' or name == 'layer3' or name == 'layer4' or name == 'avgpool':
                    layers.append(module)
                else:
                    continue
            layers = nn.Sequential(*layers)
            self.individual_extractors.append(layers)
        self.individual_extractors = nn.ModuleList(self.individual_extractors)

        # individual classifiers
        self.classifiers = nn.ModuleList([nn.Sequential(nn.Linear(512, meta_class_size)) for _ in range(ensemble_size)])

    def forward(self, x):
        common_feature = self.common_extractor(x)
        out = []
        for i in range(self.ensemble_size):
            individual_feature = self.individual_extractors[i](common_feature).view(common_feature.size(0), -1)
            individual_classes = self.classifiers[i](individual_feature)
            out.append(individual_classes)
        out = torch.stack(out, dim=1)
        return out
