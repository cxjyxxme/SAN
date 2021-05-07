import torch
import torch.nn as nn
import global_args
import math
import torchvision
from lib.sa.modules import Subtraction, Subtraction2, Aggregation
# from DCNv2.dcn_v2 import DCN
from dcn import DFConv2d
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F

# when training, do relu, return [relu(random()+x), relu(x)]
# when testing, as normal BatchNorm2d
class RandomBatchNorm2d(nn.BatchNorm2d): 
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, random_range=0, fix_rbn=False):
        super(RandomBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.random_range = random_range
        self.relu = nn.ReLU(inplace=True)
        self.fix_rbn = fix_rbn
        self.world_size = dist.get_world_size()

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        if (self.training):
            input_ = input
            batchsize, channels, height, width = input_.size()
            numel = batchsize * height * width
            input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
            sum_ = input_.sum(1)
            sum_of_square = input_.pow(2).sum(1)
            if (self.fix_rbn):
                all_reduce(sum_, op=dist.ReduceOp.SUM)
                all_reduce(sum_of_square, op=dist.ReduceOp.SUM)
                numel = numel * self.world_size
            mean = sum_ / numel
            sumvar = sum_of_square - sum_ * mean

            self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean.detach()
            )
            unbias_var = sumvar / (numel - 1)
            self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * unbias_var.detach()
            )

            bias_var = sumvar / numel
            inv_std = 1 / (bias_var + self.eps).pow(0.5)
            output = (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1)

            # sum__ = output.sum(1)
            # sum_of_square_ = output.pow(2).sum(1)
            # mean_ = sum__ / numel
            # sumvar_ = sum_of_square_ - sum__ * mean_
            # unbias_var_ = sumvar_ / (numel - 1)
            # if (input_.device==torch.device('cuda:0')):
            #     print(input_.device)
            #     print(input_.shape)
            #     print(output.shape)
            #     print('mean:',mean_.mean())
            #     print('var:',unbias_var_.mean())
            rand = torch.randn(output.shape, device=output.device) * self.random_range
            output2 = (output + rand) #/ math.sqrt(1 + self.random_range * self.random_range)
            sum__ = output2.sum(1)
            sum_of_square_ = output2.pow(2).sum(1)
            if (self.fix_rbn):
                all_reduce(sum__, op=dist.ReduceOp.SUM)
                all_reduce(sum_of_square_, op=dist.ReduceOp.SUM)
            mean_ = sum__ / numel
            sumvar_ = sum_of_square_ - sum__ * mean_
            unbias_var_ = sumvar_ / (numel - 1)
            bias_var_ = sumvar_ / numel
            inv_std_ = 1 / (bias_var_ + self.eps).pow(0.5)
            output2 = (output2 - mean_.unsqueeze(1)) * inv_std_.unsqueeze(1)
            # if (input_.device==torch.device('cuda:0')):
            #     print('mean2:',mean_.mean())
            #     print('var2:',unbias_var_.mean())
            # sum__ = output2.sum(1)
            # sum_of_square_ = output2.pow(2).sum(1)
            # mean_ = sum__ / numel
            # sumvar_ = sum_of_square_ - sum__ * mean_
            # unbias_var_ = sumvar_ / (numel - 1)
            # if (input_.device==torch.device('cuda:0')):
            #     print('mean3:',mean_.mean())
            #     print('var3:',unbias_var_.mean())

            output = (output * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)).view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()
            output2 = (output2 * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)).view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()
            return [self.relu(output), self.relu(output2)]
        else:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

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
            self.hn = out_planes // share_planes
            if (self.args.use_mask):
                self.conv_mask = nn.Conv2d(in_planes, out_planes // share_planes, kernel_size=1, stride=stride)
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
        self.weight = None

    def forward(self, x):
        if (isinstance(x, list)):
            bs, c, h, w = x[0].shape
        else:
            bs, c, h, w = x.shape
        if (self.args.new_add_random and self.training):
            x_norand = x[0]
            x_rand = x[1]
            x1 = x_rand
            x2 = x_rand
            x3 = x_norand
            x1, x2, x3 = self.conv1(x1), self.conv2(x2), self.conv3(x3) 
            x = x[0]   
        elif (self.args.add_random and self.training):
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
            if (self.args.use_mask):
                mask = torch.sigmoid(self.conv_mask(x)).reshape([bs, self.hn, 1, -1])
                w = w * mask
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        self.weight = w
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
        if (self.args.new_add_random):
            self.bn1 = RandomBatchNorm2d(in_planes, random_range=self.args.add_random_size, fix_rbn=self.args.sbn)
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        if (self.training and (self.args.add_random or self.args.new_add_random)):
            out = self.bn1(x)
        else:
            out = self.relu(self.bn1(x))
        # if (self.args.add_random and self.training):
        #     out = self.bn1(x)
        # else:
        #     out = self.relu(self.bn1(x))
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

def get_sam(planes, stride):
    return SAM(sa_type=0, in_planes=planes, rel_planes=planes//16, out_planes=planes, share_planes=8, kernel_size=7, stride=stride, dilation=1)

def resnet_san(use_sam_stages):
    args = global_args.get_args()
    assert(not args.add_random)
    if (args.resnet_type == '50'):
        model = torchvision.models.resnet50(pretrained=False)
    elif (args.resnet_type == '101'):
        model = torchvision.models.resnet101(pretrained=False)
    #TODO FIX RELU before sam
    for i in range(4):
        if not use_sam_stages[i]:
            continue
        layer = getattr(model, 'layer'+str(i+1))
        for j in range(len(layer)):
            if (args.down_sample_only and j != 0):
                continue
            if (args.use_dcn):
                stride = layer[j].conv2.stride
                assert stride[0]==stride[-1],"stride must be same"
                stride = stride[0]
                layer[j].conv2 = DFConv2d(
                    in_channels=layer[j].conv2.in_channels,
                    out_channels=layer[j].conv2.in_channels,
                    stride=stride)
            else:
                layer[j].conv2 = get_sam(layer[j].conv2.in_channels, layer[j].conv2.stride)
    return model

if __name__ == '__main__':
    net = san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=1000).cuda().eval()
    print(net)
    y = net(torch.randn(4, 3, 224, 224).cuda())
    print(y.size())
