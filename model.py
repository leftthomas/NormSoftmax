import torch.nn as nn
from capsule_layer import CapsuleLinear
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if name == 'fc':
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        # classifier
        self.in_length = 16
        self.fc = CapsuleLinear(out_capsules=num_class, in_length=self.in_length, out_length=8, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        feature = x.view(x.size(0), -1, self.in_length)
        out = self.fc(feature)
        return out
