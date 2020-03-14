import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50


class ProxyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProxyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        output = x.matmul(F.normalize(self.weight, dim=-1).t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class Model(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()

        self.feature = []
        for name, module in resnet50(pretrained=True).named_children():
            if isinstance(module, nn.Linear):
                continue
            self.feature.append(module)
        self.feature = nn.Sequential(*self.feature)

        # Refactor Layer
        self.refactor = nn.Linear(2048, feature_dim)
        # Classification Layer
        self.fc = ProxyLinear(feature_dim, num_classes)

    def forward(self, x):
        feature = self.feature(x)
        global_feature = torch.flatten(feature, start_dim=1)
        global_feature = F.layer_norm(global_feature, [global_feature.size(-1)])
        feature = F.normalize(self.refactor(global_feature), dim=-1)
        classes = self.fc(feature)
        return feature, classes
