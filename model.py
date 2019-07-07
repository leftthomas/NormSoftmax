import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, meta_class_size, ensemble_size):
        super(Model, self).__init__()

        # backbone
        self.features = []
        for _ in range(ensemble_size):
            basic_model, layers = resnet18(pretrained=True), []
            for name, module in basic_model.named_children():
                if name == 'fc':
                    continue
                layers.append(module)
            self.features.append(nn.Sequential(*layers))
        self.features = nn.ModuleList(self.features)

        # classifier
        self.fcs = nn.ModuleList([nn.Sequential(nn.Linear(512, meta_class_size)) for _ in range(ensemble_size)])

    def forward(self, x):
        feature = []
        for feature_extrator in self.features:
            feature.append(feature_extrator(x).view(x.size(0), -1))
        out = [fc(feature[index]) for index, fc in enumerate(self.fcs)]
        out = torch.stack(out, dim=1)
        return out
