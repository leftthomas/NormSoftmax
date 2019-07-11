import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class GridAttentionBlock(nn.Module):
    r"""Applies an grid attention over an input signal
    Reference papers
    Attention-Gated Networks https://arxiv.org/abs/1804.05338 & https://arxiv.org/abs/1808.08114
    Reference code
    https://github.com/ozan-oktay/Attention-Gated-Networks
    Args:
        in_features_l (int): Number of channels in the input tensor
        in_features_g (int): Number of channels in the output tensor
        attn_features (int): Number of channels in the middle tensor
        scale_factor (int): up sample factor
    """

    def __init__(self, in_features_l, in_features_g, attn_features, scale_factor=2):
        super(GridAttentionBlock, self).__init__()
        attn_features = attn_features if attn_features > 0 else 1

        self.W_l = nn.Conv2d(in_features_l, attn_features, scale_factor, scale_factor, bias=False)
        self.W_g = nn.Conv2d(in_features_g, attn_features, 1, 1, bias=True)
        self.psi = nn.Conv2d(attn_features, 1, 1, 1, bias=True)

        # output transform
        self.W = nn.Sequential(nn.Conv2d(in_features_l, in_features_l, 1, 1, bias=True), nn.BatchNorm2d(in_features_l))

    def forward(self, l, g):
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        g_ = F.interpolate(g_, size=l_.size()[2:], mode='bilinear', align_corners=False)
        c = self.psi(F.relu(l_ + g_))
        # compute attention map
        a = torch.sigmoid(c)
        a = F.interpolate(a, size=l.size()[2:], mode='bilinear', align_corners=False)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l)
        f = self.W(f)
        return f


class Model(nn.Module):

    def __init__(self, meta_classes, ensemble_size):
        super(Model, self).__init__()

        # backbone
        self.meta_classes, self.ensemble_size = meta_classes, ensemble_size

        # common features
        basic_model, self.common_extractor = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if name != 'layer3' and name != 'layer4' and name != 'avgpool' and name != 'fc':
                self.common_extractor.append(module)
            else:
                continue
        self.common_extractor = nn.Sequential(*self.common_extractor)

        # sole features
        self.sole_extractors = []
        for _ in range(ensemble_size):
            basic_model, layers = resnet18(pretrained=True), []
            for name, module in basic_model.named_children():
                if name == 'layer3' or name == 'layer4':
                    layers.append(module)
                else:
                    continue
            layers = nn.Sequential(*layers)
            self.sole_extractors.append(layers)
        self.sole_extractors = nn.ModuleList(self.sole_extractors)

        # attention block
        self.sole_attentions = nn.ModuleList([GridAttentionBlock(128, 512, 128 // 2) for _ in range(ensemble_size)])
        self.sole_w1 = nn.ModuleList([nn.Linear(8192, 512) for _ in range(ensemble_size)])
        self.sole_w2 = nn.ModuleList([nn.Linear(2048, 512) for _ in range(ensemble_size)])

        # sole classifiers
        self.classifiers = nn.ModuleList([nn.Sequential(nn.Linear(1024, meta_classes)) for _ in range(ensemble_size)])

    def forward(self, x):
        common_feature = self.common_extractor(x)
        out = []
        for i in range(self.ensemble_size):
            sole_feature = self.sole_extractors[i](common_feature)
            att_feature = self.sole_attentions[i](common_feature, sole_feature)
            sole_feature = F.adaptive_avg_pool2d(sole_feature, output_size=4).view(sole_feature.size(0), -1)
            att_feature = F.adaptive_avg_pool2d(att_feature, output_size=4).view(att_feature.size(0), -1)
            sole_feature = self.sole_w1[i](sole_feature)
            att_feature = self.sole_w2[i](att_feature)
            mix_feature = torch.cat((sole_feature, att_feature), dim=-1)
            sole_classes = self.classifiers[i](mix_feature)
            out.append(sole_classes)
        out = torch.stack(out, dim=1)
        return out
