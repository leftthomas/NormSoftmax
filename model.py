import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, meta_class_size, ensemble_size):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if name == 'fc':
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        # classifier
        self.fcs = nn.ModuleList([nn.Linear(512, meta_class_size, bias=True) for _ in range(ensemble_size)])

    def forward(self, x):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        out = [fc(feature) for fc in self.fcs]
        out = torch.stack(out, dim=1)
        return out
