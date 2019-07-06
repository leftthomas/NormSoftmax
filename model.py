import torch
import torch.nn as nn
from capsule_layer import CapsuleLinear
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, meta_class_size, ensemble_size):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if name == 'fc' or name == 'avgpool':
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        # classifier
        self.fcs = nn.ModuleList(
            [nn.Sequential(CapsuleLinear(32, 512, 32), CapsuleLinear(meta_class_size, 32, 8)) for _ in
             range(ensemble_size)])

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        feature = x.view(x.size(0), -1, 512)
        out = [fc(feature).norm(dim=-1) for fc in self.fcs]
        out = torch.stack(out, dim=1)
        return out
