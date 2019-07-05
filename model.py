import torch.nn as nn
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
        self.fc = nn.Linear(in_features=512, out_features=num_class, bias=True)

    def forward(self, x):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        out = self.fc(feature)
        return out
