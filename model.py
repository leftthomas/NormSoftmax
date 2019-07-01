import torch.nn as nn
from capsule_layer import CapsuleLinear
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, num_class, classifier_type):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if name == 'fc':
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        # classifier
        self.classifier_type = classifier_type
        if self.classifier_type == 'linear':
            self.fc = nn.Linear(512, num_class)
        else:
            self.in_length = 16
            self.fc = CapsuleLinear(out_capsules=num_class, in_length=self.in_length, out_length=8, squash=False)

    def forward(self, x):
        x = self.features(x)
        if self.classifier_type == 'linear':
            feature = x.view(x.size(0), -1)
            out = self.fc(feature)
        else:
            x = x.permute(0, 2, 3, 1).contiguous()
            feature = x.view(x.size(0), -1, self.in_length)
            out = self.fc(feature)
            out = out.norm(dim=-1)
        return out
