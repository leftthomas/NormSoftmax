import torch
import torch.nn as nn
from torchvision.models.resnet import resnext50_32x4d


class Model(nn.Module):

    def __init__(self, meta_class_size, ensemble_size, device_ids):
        super(Model, self).__init__()

        # configs
        self.meta_class_size, self.ensemble_size, self.device_ids = meta_class_size, ensemble_size, device_ids

        # common features
        basic_model, self.common_extractor = resnext50_32x4d(pretrained=True), []
        for name, module in basic_model.named_children():
            if name == 'conv1' or name == 'bn1' or name == 'relu' or name == 'maxpool' or name == 'layer1':
                self.common_extractor.append(module)
            else:
                continue
        self.common_extractor = nn.Sequential(*self.common_extractor).cuda(device_ids[0])

        # individual features
        self.individual_extractors = []
        for i in range(ensemble_size):
            basic_model, layers = resnext50_32x4d(pretrained=True), []
            for name, module in basic_model.named_children():
                if name == 'layer2' or name == 'layer3' or name == 'layer4' or name == 'avgpool':
                    layers.append(module)
                else:
                    continue
            layers = nn.Sequential(*layers)
            self.individual_extractors.append(layers)
        self.individual_extractors = nn.ModuleList(self.individual_extractors).cuda(device_ids[1])

        # individual classifiers
        self.classifiers = nn.ModuleList(
            [nn.Sequential(nn.Linear(512 * 4, meta_class_size)) for _ in range(ensemble_size)]).cuda(device_ids[0])

    def forward(self, x):
        common_feature = self.common_extractor(x)
        common_feature = common_feature.cuda(self.device_ids[1])
        out = []
        for i in range(self.ensemble_size):
            individual_feature = self.individual_extractors[i](common_feature).view(common_feature.size(0), -1)
            individual_feature = individual_feature.cuda(self.device_ids[0])
            individual_classes = self.classifiers[i](individual_feature)
            out.append(individual_classes)
        out = torch.stack(out, dim=1)
        return out
