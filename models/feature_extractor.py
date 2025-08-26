import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# It contains feature extractor
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # add feature extractor!
        self.convA = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.convB = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.convA(x)
        # x = self.tanh(x)
        # x = self.convB(x)
        feature = self.tanh(x)  # features extracted after extractor
        # print(feature.shape)
        x = self.conv1(feature)
        # print(f"After layer1: {mid_feature.shape}")
        x = self.bn1(x)
        out = F.relu(x)
        # print(f"After conv1: {out.shape}")
        out = self.maxpool(out)
        # print(out.shape)
        out = self.layer1(out)
        # print(f"After layer1: {out.shape}")
        out = self.layer2(out)
        # print(f"After layer2: {out.shape}")
        feature_new = self.layer3(out)
        # print(f"After layer3: {out.shape}")
        out = self.layer4(feature_new)
        # print(f"After layer4: {feature.shape}")
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # print(f"After adaptive_avg_pool2d: {out.shape}")
        out = out.view(out.size(0), -1)
        # print(f"After flatten: {out.shape}")
        out = self.linear(out)
        # print(f"Output shape: {out.shape}")

        if return_features:
            return out, feature, feature_new
        else:
            return out


def resnet18_new(num_classes=10, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)

