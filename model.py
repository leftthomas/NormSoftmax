import torch
import torch.nn as nn
from capsule_layer import CapsuleLinear
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, meta_class_size, ensemble_size):
        super(Model, self).__init__()

        # backbone
        self.meta_class_size, self.ensemble_size = meta_class_size, ensemble_size

        # common features
        basic_model, self.common_extractor = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if name != 'layer3' and name != 'layer4' and name != 'avgpool' and name != 'fc':
                self.common_extractor.append(module)
            else:
                continue
        self.common_extractor = nn.Sequential(*self.common_extractor)

        # individual features
        self.individual_extractors = []
        for _ in range(ensemble_size):
            basic_model, layers = resnet18(pretrained=True), []
            for name, module in basic_model.named_children():
                if name == 'layer3' or name == 'layer4':
                    layers.append(module)
                else:
                    continue
            layers = nn.Sequential(*layers)
            self.individual_extractors.append(layers)
        self.individual_extractors = nn.ModuleList(self.individual_extractors)

        # capsule cluster
        self.individual_clusters = nn.ModuleList(
            [CapsuleLinear(1, 512, 512, num_iterations=3, bias=False, squash=False) for _ in range(ensemble_size)])

        # individual classifiers
        self.classifiers = nn.ModuleList([nn.Sequential(nn.Linear(512, meta_class_size)) for _ in range(ensemble_size)])

    def forward(self, x):
        common_feature = self.common_extractor(x)
        out = []
        for i in range(self.ensemble_size):
            individual_feature = self.individual_extractors[i](common_feature).permute(0, 2, 3, 1).contiguous()
            individual_feature = individual_feature.view(individual_feature.size(0), -1, 512)
            individual_cluster = self.individual_clusters[i](individual_feature).permute(0, 2, 1).contiguous()
            individual_cluster = individual_cluster.view(individual_cluster.size(0), -1)
            individual_classes = self.classifiers[i](individual_cluster)
            out.append(individual_classes)
        out = torch.stack(out, dim=1)
        return out
