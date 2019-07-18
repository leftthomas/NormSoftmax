import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.layer2, self.layer3, self.layer4 = [], [], []
        for i in range(ensemble_size):
            basic_model = resnext50_32x4d(pretrained=True)
            for name, module in basic_model.named_children():
                if name == 'layer2':
                    self.layer2.append(module)
                if name == 'layer3':
                    self.layer3.append(module)
                if name == 'layer4':
                    self.layer4.append(module)
                else:
                    continue
        self.layer2 = nn.ModuleList(self.layer2).cuda(device_ids[0])
        self.layer3 = nn.ModuleList(self.layer3).cuda(device_ids[1])
        self.layer4 = nn.ModuleList(self.layer4).cuda(device_ids[2])

        # individual classifiers
        self.classifiers = nn.ModuleList(
            [nn.Sequential(nn.Linear(512 * 4, meta_class_size)) for _ in range(ensemble_size)]).cuda(device_ids[2])

    def forward(self, x):
        common_feature = self.common_extractor(x)
        out = []
        for i in range(self.ensemble_size):
            individual_feature = self.layer2[i](common_feature)
            individual_feature = individual_feature.cuda(self.device_ids[1])
            individual_feature = self.layer3[i](individual_feature)
            individual_feature = individual_feature.cuda(self.device_ids[2])
            individual_feature = self.layer4[i](individual_feature)
            individual_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1))
            individual_feature = individual_feature.view(individual_feature.size(0), -1)
            individual_classes = self.classifiers[i](individual_feature)
            out.append(individual_classes)
        out = torch.stack(out, dim=1)
        return out
