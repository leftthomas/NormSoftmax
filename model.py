import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnext50_32x4d


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, with_se=False):
        super(AttentionBlock, self).__init__()
        self.up_factor, self.with_se = up_factor, with_se
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
        if with_se:
            self.channel_gate = SEBlock(in_features_l)

    def forward(self, l, g):
        # re-weight local channel feature
        if self.with_se:
            l = self.channel_gate(l)
        l_ = self.W_l(l)
        # gate feature up-sample
        g_ = self.W_g(g)
        g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bicubic', align_corners=False)
        # c is heat map
        c = self.phi(F.relu(l_ + g_))
        # compute spatial attention map
        a = torch.sigmoid(c)
        # re-weight local spatial feature
        f = torch.mul(a, l)
        return f


class Model(nn.Module):
    def __init__(self, meta_class_size, ensemble_size, model_type, with_se, device_ids):
        super(Model, self).__init__()

        # backbone
        backbones = {'resnet18': (resnet18, 1), 'resnet34': (resnet34, 1), 'resnet50': (resnet50, 4),
                     'resnext50_32x4d': (resnext50_32x4d, 4)}
        backbone, expansion = backbones[model_type]

        # configs
        self.ensemble_size, self.device_ids = ensemble_size, device_ids

        # common features
        basic_model, self.common_extractor = backbone(pretrained=True), []
        for name, module in basic_model.named_children():
            if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']:
                self.common_extractor.append(module)
            else:
                continue
        self.common_extractor = nn.Sequential(*self.common_extractor).cuda(device_ids[0])

        # individual features
        self.layer2, self.layer3, self.layer4, self.attention1, self.attention2 = [], [], [], [], []
        for i in range(ensemble_size):
            basic_model = backbone(pretrained=True)
            for name, module in basic_model.named_children():
                if name == 'layer2':
                    self.layer2.append(module)
                if name == 'layer3':
                    self.layer3.append(module)
                if name == 'layer4':
                    self.layer4.append(module)
                else:
                    continue
            self.attention1.append(AttentionBlock(128, 512, 128, 4, with_se=with_se))
            self.attention2.append(AttentionBlock(256, 512, 256, 2, with_se=with_se))
        self.layer2 = nn.ModuleList(self.layer2).cuda(device_ids[0])
        self.layer3 = nn.ModuleList(self.layer3).cuda(device_ids[1])
        self.layer4 = nn.ModuleList(self.layer4).cuda(device_ids[1])
        self.attention1 = nn.ModuleList(self.attention1).cuda(device_ids[2])
        self.attention2 = nn.ModuleList(self.attention2).cuda(device_ids[2])

        # individual classifiers
        self.classifiers = nn.ModuleList([nn.Sequential(nn.Linear((128 + 256 + 512) * expansion, meta_class_size)) for _
                                          in range(ensemble_size)]).cuda(device_ids[2])

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.common_extractor(x)
        out = []
        for i in range(self.ensemble_size):
            layer2_feature = self.layer2[i](common_feature)
            layer3_feature = self.layer3[i](layer2_feature.cuda(self.device_ids[1]))
            layer4_feature = self.layer4[i](layer3_feature)
            g1 = self.attention1[i](layer2_feature.cuda(self.device_ids[2]), layer4_feature.cuda(self.device_ids[2]))
            g2 = self.attention2[i](layer3_feature.cuda(self.device_ids[2]), layer4_feature.cuda(self.device_ids[2]))
            g1 = F.adaptive_max_pool2d(g1, output_size=(1, 1)).view(batch_size, -1)
            g2 = F.adaptive_max_pool2d(g2, output_size=(1, 1)).view(batch_size, -1)
            g3 = F.adaptive_max_pool2d(layer4_feature.cuda(self.device_ids[2]), output_size=(1, 1)).view(batch_size, -1)
            global_feature = torch.cat([g1, g2, g3], dim=-1)
            classes = self.classifiers[i](global_feature)
            out.append(classes)
        out = torch.stack(out, dim=1)
        return out
