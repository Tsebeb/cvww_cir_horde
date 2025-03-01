import torch.nn as nn

from networks.utils import CosineLinear


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

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
        if not self.last: #remove ReLU in the last layer
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, nf=64):
        self.inplanes = nf
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, nf, layers[0])
        self.layer2 = self._make_layer(block, nf * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, layers[3], stride=2, last_phase=True)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = CosineLinear(nf * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head_var = "fc"

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_lucir(num_classes):
    """Constructs a ResNet-18 model with a CosineSimilarityHead """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def slimresnet18_lucir(num_classes):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, nf=20)
    return model
