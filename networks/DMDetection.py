import torch
import torch.nn as nn


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class ChannelLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, pool = None) -> None:
        super(ChannelLinear, self).__init__(in_features, out_features, bias)
        self.compute_axis = 1
        self.pool = pool

    def forward(self, x):
        axis_ref = len(x.shape)-1
        x = torch.transpose(x, self.compute_axis, axis_ref)
        out_shape = list(x.shape); out_shape[-1] = self.out_features
        x = x.reshape(-1, x.shape[-1])
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None,:]
        x = torch.transpose(x.view(out_shape), axis_ref, self.compute_axis)
        if self.pool is not None:
            x = self.pool(x)
        return x

    
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.padding==0:
            identity = identity[...,1:-1,1:-1]
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
        

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, stride0=2,
                 padding=1, dropout=0.0, gap_size=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=stride0, padding=3*padding, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if dropout>0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride0, padding=padding)
        self.layer1 = self._make_layer(block, 64, layers[0], padding=padding)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, padding=padding)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, padding=padding)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, padding=padding)
        
        if gap_size is None:
            self.gap_size = None
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif gap_size<0:
            with torch.no_grad():
                y = self.feature(torch.zeros((1,3,-gap_size,-gap_size), dtype=torch.float32)).shape
            print('gap_size:', -gap_size, '>>', y[-1])
            self.gap_size = y[-1]
            self.avgpool = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        elif gap_size==1:
            self.gap_size = gap_size
            self.avgpool = None
        else:
            self.gap_size = gap_size
            self.avgpool = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        self.num_features = 512 * block.expansion
        self.fc = ChannelLinear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, padding=padding))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=padding))
        return nn.Sequential(*layers)


    def feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        x = self.feature(x)
        if self.avgpool is not None:
            x = self.avgpool(x)
        if self.dropout is not None:    
            x = self.dropout(x)
        y = self.fc(x)


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model