import torch
import torch.nn as nn
import global_args
import math

from lib.sa.modules import Subtraction, Subtraction2, Aggregation


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.args = global_args.get_args()
        self.ws = out_planes // share_planes
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if (self.args.use_position2):
            ch = share_planes + kernel_size * kernel_size
            self.conv_pos2 = nn.Conv2d((ch) * (out_planes // share_planes), out_planes, kernel_size=1, groups=out_planes // share_planes, bias=False)

        self.relu = nn.ReLU(inplace=True)
        if sa_type == 0:
            if (self.args.use_position):
                self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
                ch = rel_planes + 2
            else:
                ch = rel_planes
            self.conv_w = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                                        nn.Conv2d(ch, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        if (self.args.add_random and self.training):
            rand = torch.randn(x.shape, device=x.device) * self.args.add_random_size
            x_ = (x + rand) / math.sqrt(1 + self.args.add_random_size * self.args.add_random_size)
            x_ = self.relu(x_)
            x1 = self.conv1(x_)   #[bs, k_ch, h, w]
            x2 = self.conv2(x_)
            x = self.relu(x)
            x3 = self.conv3(x)
        else:
            x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)

        if self.sa_type == 0:  # pairwise
            if (self.args.use_position):
                p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
                w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
            else:
                w = self.softmax(self.conv_w(self.subtraction2(x1, x2)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        if (self.args.use_position2):
            x = x.view(x.shape[0], x.shape[1] // w.shape[1], w.shape[1], x.shape[2], x.shape[3]).permute([0,2,1,3,4])
            x = torch.cat([x, w.view(w.shape[0], w.shape[1], w.shape[2], x.shape[3], x.shape[4])], 2).view(x.shape[0], -1, x.shape[3], x.shape[4])
            x = self.conv_pos2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.args = global_args.get_args()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        if (self.args.add_random and self.training):
            out = self.bn1(x)
        else:
            out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out


class SAN(nn.Module):
    def __init__(self, sa_type, block, layers, kernels, num_classes):
        super(SAN, self).__init__()
        self.args = global_args.get_args()
        c = 64
        self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.layer0 = self._make_layer(sa_type, block, c, layers[0], kernels[0])

        c *= 4
        self.conv1, self.bn1 = conv1x1(c // 4, c), nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(sa_type, block, c, layers[1], kernels[1])

        c *= 2
        self.conv2, self.bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer2 = self._make_layer(sa_type, block, c, layers[2], kernels[2])

        c *= 2
        self.conv3, self.bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer3 = self._make_layer(sa_type, block, c, layers[3], kernels[3])

        c *= 2
        self.conv4, self.bn4 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer4 = self._make_layer(sa_type, block, c, layers[4], kernels[4])

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, num_classes)

    def _make_layer(self, sa_type, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def san(sa_type, layers, kernels, num_classes):
    model = SAN(sa_type, Bottleneck, layers, kernels, num_classes)
    return model


if __name__ == '__main__':
    net = san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=1000).cuda().eval()
    print(net)
    y = net(torch.randn(4, 3, 224, 224).cuda())
    print(y.size())
