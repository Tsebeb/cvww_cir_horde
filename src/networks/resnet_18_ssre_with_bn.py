import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class conv_block(nn.Module):

    def __init__(self, in_planes, planes, mode, stride=1):
        super(conv_block, self).__init__()
        self.conv = conv3x3(in_planes, planes, stride)
        self.mode = mode
        if mode == 'parallel_adapters':
            self.adapter = conv1x1(in_planes, planes, stride)

    def forward(self, x):
        y = self.conv(x)
        if self.mode == 'parallel_adapters':
            # x = F.dropout2d(x, p=0.5, training=self.training)
            y = y + self.adapter(x)

        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, mode, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, mode, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_block(planes, planes, mode)
        self.bn2 = nn.BatchNorm2d(planes)
        self.mode = mode

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, nf=64, mode='normal'):
        self.inplanes = nf
        super(ResNet, self).__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf, layers[0])
        self.layer2 = self._make_layer(block, nf * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.feature_dim = nf * 8
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.head_var = "fc"

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.mode, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.mode))

        return nn.Sequential(*layers)

    def switch(self, mode='normal'):
        for name, module in self.named_modules():
            if hasattr(module, 'mode'):
                module.mode = mode

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        dim = x.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        x = pool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet18_ssre_bn(pretrained, mode="parallel_adapters", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], mode=mode, **kwargs)
    return model


def slimresnet18_ssre_bn(pretrained, mode="parallel_adapters", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], nf=20, mode=mode, **kwargs)
    return model

