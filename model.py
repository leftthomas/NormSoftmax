import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnext50_32x4d


class Model(nn.Module):
    def __init__(self, meta_class_size, ensemble_size, share_type, model_type, with_random, device_ids):
        super(Model, self).__init__()

        # backbone
        backbones = {'resnet18': (resnet18, 1), 'resnet34': (resnet34, 1), 'resnet50': (resnet50, 4),
                     'resnext50_32x4d': (resnext50_32x4d, 4)}
        backbone, expansion = backbones[model_type]
        module_names, modules = list(zip(*list(backbone(pretrained=True).named_children())))
        module_names, modules = module_names[: -2], modules[: -2]

        # configs
        self.ensemble_size, self.with_random, self.device_ids = ensemble_size, with_random, device_ids

        # common features
        self.common_extractor = copy.deepcopy(nn.Sequential(*modules[:module_names.index(share_type) + 1])) \
            .cuda(device_ids[0])
        print("# trainable common feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.common_extractor.parameters()))

        # individual features
        self.individual_extractor = nn.ModuleList(
            [copy.deepcopy(nn.Sequential(*modules[module_names.index(share_type) + 1:])).cuda(device_ids[1]) for _
             in range(ensemble_size)])
        print("# trainable individual feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.individual_extractor.parameters())
              // ensemble_size)

        # individual classifiers
        self.classifiers = nn.ModuleList([nn.Linear(512 * expansion, meta_class_size).cuda(device_ids[1])
                                          for _ in range(ensemble_size)])
        print("# trainable individual classifier parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.classifiers.parameters()) // ensemble_size)

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.common_extractor(x)
        if self.with_random:
            branch_weight = torch.rand(self.ensemble_size, device=x.device)
            branch_weight = F.softmax(branch_weight, dim=-1)
        else:
            branch_weight = torch.ones(self.ensemble_size, device=x.device)
        out = []
        for i in range(self.ensemble_size):
            individual_feature = self.individual_extractor[i](
                (branch_weight[i] * common_feature).cuda(self.device_ids[1]))
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            classes = self.classifiers[i](global_feature)
            out.append(classes)
        out = torch.stack(out, dim=1)
        return out
