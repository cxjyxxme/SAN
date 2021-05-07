import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.distributed as dist
from lib.sa.modules import Subtraction, Subtraction2, Aggregation
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

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
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1,use_position=False,use_position2=True,add_random=True,add_random_size=0.02,new_add_random=False):
        super(SAM, self).__init__()
        self.args = {
            "use_position": use_position,
            "use_position2": use_position2,
            "add_random": add_random,
            "add_random_size": add_random_size,
            "new_add_random": new_add_random
        }
        self.random_qkv = [True, True, False]
        self.ws = out_planes // share_planes
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if self.args["use_position2"]:
            ch = share_planes + kernel_size * kernel_size
            self.conv_pos2 = nn.Conv2d((ch) * (out_planes // share_planes), out_planes, kernel_size=1, groups=out_planes // share_planes, bias=False)
        # change
        self.relu = nn.ReLU(inplace=False)
        if sa_type == 0:
            if self.args["use_position"]:
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
        if (self.args["new_add_random"] and self.training):
            xshape = x[0].shape
            x_norand = x[0]
            x_rand = x[1]
            if (self.random_qkv[0]):
                x1 = x_rand
            else:
                x1 = x_norand
            if (self.random_qkv[1]):
                x2 = x_rand
            else:
                x2 = x_norand
            if (self.random_qkv[2]):
                x3 = x_rand
            else:
                x3 = x_norand
            # x1 = self.conv1(x[0])   #[bs, k_ch, h, w]
            # x2 = self.conv2(x[0])
            # x3 = self.conv3(x[1])
        elif (self.args["add_random"] and self.training):
            xshape = x.shape
            rand = torch.randn(x.shape, device=x.device) * self.args["add_random_size"]
            x_ = (x + rand) / math.sqrt(1 + self.args["add_random_size"] * self.args["add_random_size"])
            x_ = self.relu(x_)
            # x1 = self.conv1(x_)   #[bs, k_ch, h, w]
            # x2 = self.conv2(x_)
            x = self.relu(x)
            # x3 = self.conv3(x)
            if (self.random_qkv[0]):
                x1 = x_
            else:
                x1 = x
            if (self.random_qkv[1]):
                x2 = x_
            else:
                x2 = x
            if (self.random_qkv[2]):
                x3 = x_
            else:
                x3 = x
        else:
            xshape = x.shape
            x1, x2, x3 = x, x, x
        x1, x2, x3 = self.conv1(x1), self.conv2(x2), self.conv3(x3)    
        if self.sa_type == 0:  # pairwise
            if (self.args["use_position"]):
                p = self.conv_p(position(xshape[2], xshape[3], x1.is_cuda))
                w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(xshape[0], 1, 1, 1)], 1)))
            else:
                w = self.softmax(self.conv_w(self.subtraction2(x1, x2)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(xshape[0], -1, 1, xshape[2]*xshape[3])
            x2 = self.unfold_j(self.pad(x2)).view(xshape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(xshape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        if (self.args["use_position2"]):
            x = x.view(x.shape[0], x.shape[1] // w.shape[1], w.shape[1], x.shape[2], x.shape[3]).permute([0,2,1,3,4])
            x = torch.cat([x, w.view(w.shape[0], w.shape[1], w.shape[2], x.shape[3], x.shape[4])], 2).view(x.shape[0], -1, x.shape[3], x.shape[4])
            x = self.conv_pos2(x)
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_sam=False,
        cfg=None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if (cfg.new_add_random and use_sam):
            self.bn1 = RandomBatchNorm2d(width, random_range=cfg.add_random_size, fix_rbn=cfg.sbn)
        else:
            self.bn1 = norm_layer(width)
        if (use_sam):
            self.conv2 = get_sam(cfg, width, stride)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_sam = use_sam
        self.cfg = cfg

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if (self.training and self.use_sam and (self.cfg.add_random or self.cfg.new_add_random)):
            pass
        else:
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        cfg=None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        use_sam_stages = cfg.stage_with_sam
        sam_dsonly = False
        self.layer1 = self._make_layer(block, 64, layers[0], use_sam=use_sam_stages[0], sam_dsonly=sam_dsonly, cfg=cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], use_sam=use_sam_stages[1], sam_dsonly=sam_dsonly, cfg=cfg)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], use_sam=use_sam_stages[2], sam_dsonly=sam_dsonly, cfg=cfg)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], use_sam=use_sam_stages[3], sam_dsonly=sam_dsonly, cfg=cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, use_sam=False, sam_dsonly=False, cfg=None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, use_sam=use_sam, cfg=cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, use_sam=(use_sam and (not sam_dsonly)), cfg=cfg))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def get_sam(cfg, planes, stride):
    use_position = cfg.use_position
    use_position2 = cfg.use_position2
    add_random = cfg.add_random
    add_random_size = cfg.add_random_size
    new_add_random = cfg.new_add_random
    return SAM(sa_type=0, in_planes=planes, rel_planes=planes//16, out_planes=planes, share_planes=8, kernel_size=7, stride=stride, dilation=1,
                use_position=use_position,
                use_position2=use_position2,
                add_random=add_random,
                add_random_size=add_random_size,
                new_add_random=new_add_random)

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def resnet_san_new(cfg):
    model = ResNet(Bottleneck, [3, 4, 23, 3], cfg=cfg)
    if (cfg.sbn):
        for name, m in model.named_modules():
            if (isinstance(m, nn.BatchNorm2d) and (not isinstance(m, RandomBatchNorm2d))):
                _set_module(model, name, nn.SyncBatchNorm.convert_sync_batchnorm(m))
    return model
