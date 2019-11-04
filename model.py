import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnext50_32x4d


class EfficientChannelAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(channel, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class Model(nn.Module):
    def __init__(self, meta_class_size, ensemble_size, share_type, model_type, with_random, device_ids):
        super(Model, self).__init__()

        # backbone
        backbones = {'resnet18': (resnet18, 1), 'resnet34': (resnet34, 1), 'resnet50': (resnet50, 4),
                     'resnext50_32x4d': (resnext50_32x4d, 4)}
        backbone, expansion = backbones[model_type]
        module_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        # configs
        self.ensemble_size, self.with_random, self.device_ids = ensemble_size, with_random, device_ids

        # common features
        self.common_extractor = []
        basic_model, common_module_names = backbone(pretrained=True), module_names[:module_names.index(share_type) + 1]
        for name, module in basic_model.named_children():
            if name in common_module_names:
                self.common_extractor.append(module)
        self.common_extractor = nn.Sequential(*self.common_extractor).cuda(device_ids[0])
        print("# trainable common feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.common_extractor.parameters()))

        if self.with_random:
            self.branch_attention = nn.ModuleList([EfficientChannelAttention(64).cuda(device_ids[0])
                                                   for _ in range(ensemble_size)])

        # individual features
        self.layer1, self.layer2, self.layer3, self.layer4 = [], [], [], []
        individual_module_names = module_names[module_names.index(share_type) + 1:]
        for i in range(ensemble_size):
            basic_model = backbone(pretrained=True)
            for name, module in basic_model.named_children():
                if name in individual_module_names and name == 'layer1':
                    self.layer1.append(module.cuda(device_ids[0]))
                if name in individual_module_names and name == 'layer2':
                    self.layer2.append(module.cuda(device_ids[0]))
                if name in individual_module_names and name == 'layer3':
                    self.layer3.append(module.cuda(device_ids[0 if i < ensemble_size / 6 else 1]))
                if name in individual_module_names and name == 'layer4':
                    self.layer4.append(module.cuda(device_ids[0 if i < ensemble_size / 6 else 1]))
        self.layer1 = nn.ModuleList(self.layer1)
        self.layer2 = nn.ModuleList(self.layer2)
        self.layer3 = nn.ModuleList(self.layer3)
        self.layer4 = nn.ModuleList(self.layer4)
        print("# trainable individual feature parameters:",
              (sum(param.numel() if param.requires_grad else 0 for param in self.layer1.parameters()) + sum(
                  param.numel() if param.requires_grad else 0 for param in self.layer2.parameters()) + sum(
                  param.numel() if param.requires_grad else 0 for param in self.layer3.parameters()) + sum(
                  param.numel() if param.requires_grad else 0 for param in self.layer4.parameters())) // ensemble_size)

        # individual classifiers
        self.classifiers = nn.ModuleList([nn.Linear(512 * expansion, meta_class_size).cuda(device_ids[1])
                                          for _ in range(ensemble_size)])
        print("# trainable individual classifier parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.classifiers.parameters()) // ensemble_size)

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.common_extractor(x)
        out = []
        for i in range(self.ensemble_size):
            if self.with_random:
                individual_feature = self.branch_attention[i](common_feature)
            else:
                individual_feature = common_feature
            if len(self.layer1) != 0:
                individual_feature = self.layer1[i](individual_feature.cuda(self.device_ids[0]))
            if len(self.layer2) != 0:
                individual_feature = self.layer2[i](individual_feature.cuda(self.device_ids[0]))
            if len(self.layer3) != 0:
                individual_feature = self.layer3[i](
                    individual_feature.cuda(self.device_ids[0 if i < self.ensemble_size / 6 else 1]))
            if len(self.layer4) != 0:
                individual_feature = self.layer4[i](
                    individual_feature.cuda(self.device_ids[0 if i < self.ensemble_size / 6 else 1]))
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            classes = self.classifiers[i](global_feature.cuda(self.device_ids[1]))
            out.append(classes)
        out = torch.stack(out, dim=1)
        return out
