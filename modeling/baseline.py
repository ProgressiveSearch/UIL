# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from .backbones.resnet import ResNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, resnet_path=None):
        super(Baseline, self).__init__()
        self.base = ResNet(last_stride)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.base.load_param(resnet_path)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.bottleneck.apply(weights_init_kaiming)
        self.num_classes = num_classes

        self.dropout = nn.Dropout(0.5)
        if num_classes != 0:
            self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)  # (b, 2048, 24, 8)
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        feat = self.dropout(feat)
        return F.normalize(feat, p=2, dim=1), F.normalize(global_feat, p=2, dim=1)

    def load_weight(self, basemodel_path):
        param_dict = torch.load(basemodel_path)
        for i in param_dict:
            if i in self.state_dict():
                self.state_dict()[i].copy_(param_dict[i])


class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, embeding_fea_size=1024, dropout=0.5, num_classes=0):
        super(self.__class__, self).__init__()
        # embedding
        self.embeding_fea_size = embeding_fea_size
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        nn.init.kaiming_normal_(self.embeding.weight, mode='fan_out')
        nn.init.constant_(self.embeding.bias, 0)
        nn.init.constant_(self.embeding_bn.weight, 1)
        nn.init.constant_(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)

        self.num_classes = num_classes
        if self.num_classes != 0:
            self.classifier = nn.Linear(2048, num_classes)
            nn.init.normal_(self.classifier.weight, std=0.001)
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, inputs):
        # eval_feas = F.normalize(inputs, p=2, dim=1)
        eval_feas = inputs
        net = self.embeding(inputs)
        net = self.embeding_bn(net)
        if self.num_classes != 0 and self.training:
            score = self.classifier(net)
            net = F.normalize(net, p=2, dim=1)
            net = self.drop(net)
            return net, eval_feas, score
        net = F.normalize(net, p=2, dim=1)
        net = self.drop(net)
        # embedding feature, test feature
        return net, eval_feas


class End2End_AvgPooling(nn.Module):
    def __init__(self, last_stride=2, dropout=0, embeding_fea_size=2048, num_classes=0):
        super().__init__()
        self.base = ResNet(last_stride)
        self.base.load_param('/home/liaoxingyu2/lxy/.cache/torch/checkpoints/resnet50-19c8e357.pth')
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pooling = AvgPooling(input_feature_size=2048, embeding_fea_size=embeding_fea_size, dropout=dropout,
                                      num_classes=num_classes)

    def forward(self, x):
        # resnet encoding
        resnet_feature = self.base(x)
        resnet_feature = self.gap(resnet_feature)

        # reshape back into (batch, samples, ...)
        resnet_feature = resnet_feature.view(resnet_feature.shape[0], -1)

        # avg pooling
        output = self.avg_pooling(resnet_feature)
        return output

    def load_weight(self, param_dict):
        for i in param_dict:
            if i in self.state_dict():
                self.state_dict()[i].data.copy_(param_dict[i].data)

    def reset_embedding_param(self):
        nn.init.kaiming_normal_(self.avg_pooling.embeding.weight, mode='fan_out')
        nn.init.constant_(self.avg_pooling.embeding.bias, 0)
        nn.init.constant_(self.avg_pooling.embeding_bn.weight, 1)
        nn.init.constant_(self.avg_pooling.embeding_bn.bias, 0)

