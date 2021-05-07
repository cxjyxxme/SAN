# from args import Args
# import global_args
# global_args._init()
# args = Args().parse()
# global_args.set_args(args)
# import jittor as jt
# import tcn_network as TNet
# import jittor.models as jtmodels
# from jittor import nn
# from tcn import TCN
# from dtcn import DTCN
import torch
import torchvision
from model.san import SAN, SAM, Bottleneck
from torch import nn
from lib.sa.modules import Subtraction, Subtraction2, Aggregation
import torch.nn.functional as F

def product_shape(shape):
    ss = 1
    for s in shape:
        ss *= s
    return ss

def flops_Conv2d(m, x):
    flops = 0
    flops_add = 0

    y = m(x)
    flops = (m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups) * y.shape[2] * y.shape[3] * m.out_channels
    flops_add += (m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups - 1) * y.shape[2] * y.shape[3] * m.out_channels
    if m.bias is not None:
        flops_add += product_shape(y.shape)
    return flops, flops_add, y

def flops_BatchNorm2d(m, x):
    flops = 0
    flops_add = 0

    n = m.num_features
    # w = self.weight / jt.sqrt(self.running_var+self.eps)
    flops += n * 2
    flops_add += n
    # b = self.bias - self.running_mean * w
    flops += n
    flops_add += n
    # norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
    flops += product_shape(x.shape)
    flops_add += product_shape(x.shape)
    return flops, flops_add, m(x)

def flops_relu(m, x):
    return 0, 0, m(x)

def flops_MaxPool2d(m, x):
    return 0, 0, m(x)

def flops_AdaptiveAvgPool2d(m, x):
    return 0, 0, m(x) #TODO?

def flops_Softmax(m, x):
    n = product_shape(x.shape)
    return n * 2, n * 2, m(x)

def flops_Linear(m, x):
    flops = 0
    flops_add = 0

    x = F.linear(x, m.weight, m.bias)
    flops += product_shape(x.shape) * m.in_features
    flops_add += product_shape(x.shape) * (m.in_features - 1)
    if m.bias is not None:
        x = x + m.bias
        flops_add += product_shape(x.shape)
    return flops, flops_add, x

def flops_Sequential(m, x):
    models = []
    for layer in m:
        models.append(layer)
    return flops_Sequential_(models, x)

def flops_Bottleneck(m, x):
    flops = 0
    flops_add = 0
    # identity = x
    # out = self.conv1(x)
    # out = self.bn1(out)
    # out = self.relu(out)
    # out = self.conv2(out)
    # out = self.bn2(out)
    # out = self.relu(out)
    # out = self.conv3(out)
    # out = self.bn3(out)
    # if (self.downsample is not None):
    #     identity = self.downsample(x)
    # out += identity
    # out = self.relu(out)
    identity = x
    flops, flops_add, out = flops_Sequential_([m.conv1, m.bn1, m.relu, m.conv2, m.bn2, m.relu, m.conv3, m.bn3], x)

    if (m.downsample is not None):
        flops, flops_add, identity = flops_any_layer(m.downsample, x, flops, flops_add)

    flops_add += product_shape(identity.shape)
    out += identity

    flops, flops_add, out = flops_any_layer(m.relu, out, flops, flops_add)
    return flops, flops_add, out

def flops_Bottleneck2(m, x):
    flops = 0
    flops_add = 0
    # identity = x
    # out = self.relu(self.bn1(x))
    # out = self.relu(self.bn2(self.sam(out)))
    # out = self.conv(out)
    # out += identity
    # return out
    identity = x
    flops, flops_add, out = flops_Sequential_([m.bn1, m.relu, m.sam, m.bn2, m.relu, m.conv], x)

    flops_add += product_shape(identity.shape)
    out += identity
    return flops, flops_add, out

def flops_matmul(a, b):
    c = jt.matmul(a, b)
    flops = product_shape(c.shape) * a.shape[-1]
    flops_add = product_shape(c.shape) * (a.shape[-1] - 1)
    return flops, flops_add, c

def flops_Subtraction(m, a):
    c = m(a)
    flops_add = product_shape(c.shape)
    return 0, flops_add, c

def flops_Subtraction2(m, a, b):
    c = m(a, b)
    flops_add = product_shape(c.shape)
    return 0, flops_add, c

def flops_Aggregation(m, a, b):
    c = m(a, b)
    n = a.shape[1] * b.shape[2] * b.shape[3]
    flops = flops_add = n
    return flops, flops_add, c

def flops_SAM(m, x):
    def position(H, W, is_cuda=True):
        if is_cuda:
            loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
            loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
        else:
            loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
            loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
        loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
        return loc
    flops = 0
    flops_add = 0
    flops, flops_add, x1 = flops_any_layer(m.conv1, x, flops, flops_add)
    flops, flops_add, x2 = flops_any_layer(m.conv2, x, flops, flops_add)
    flops, flops_add, x3 = flops_any_layer(m.conv3, x, flops, flops_add)
    
    if (m.args.use_position):
        p = position(x.shape[2], x.shape[3], x.is_cuda)
        flops, flops_add, p = flops_any_layer(m.conv_p, p, flops, flops_add)

        f, a, t1 = flops_Subtraction2(m.subtraction2, x1, x2)
        flops += f
        flops_add += a
        f, a, t2 = flops_Subtraction(m.subtraction, p)
        flops += f
        flops_add += a
        t2 = t2.repeat(x.shape[0], 1, 1, 1)
        tt = torch.cat([t1, t2], 1)
        f, a, w = flops_Sequential_([m.conv_w, m.softmax], tt)
        flops += f
        flops_add += a
        # w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
    else:
        # w = self.softmax(self.conv_w(self.subtraction2(x1, x2)))
        f, a, t1 = flops_Subtraction2(m.subtraction2, x1, x2)
        flops += f
        flops_add += a
        f, a, w = flops_Sequential_([m.conv_w, m.softmax], t1)
        flops += f
        flops_add += a
    f, a, x = flops_Aggregation(m.aggregation, x3, w)
    flops += f
    flops_add += a
    if (m.args.use_position2):
        x = x.view(x.shape[0], x.shape[1] // w.shape[1], w.shape[1], x.shape[2], x.shape[3]).permute([0,2,1,3,4])
        x = torch.cat([x, w.view(w.shape[0], w.shape[1], w.shape[2], x.shape[3], x.shape[4])], 2).view(x.shape[0], -1, x.shape[3], x.shape[4])
        flops, flops_add, x = flops_any_layer(m.conv_pos2, x, flops, flops_add)
    return flops, flops_add, x

def flops_any_layer(m, x, flops, flops_add):
    if (isinstance(m, nn.Conv2d)):
        f, a, x = flops_Conv2d(m, x)
    elif (isinstance(m, nn.BatchNorm2d)):
        f, a, x = flops_BatchNorm2d(m, x)
    elif (isinstance(m, nn.ReLU)):
        f, a, x = flops_relu(m, x)
    elif (isinstance(m, nn.MaxPool2d)):
        f, a, x = flops_MaxPool2d(m, x)
    elif (isinstance(m, nn.Sequential)):
        f, a, x = flops_Sequential(m, x)
    elif (isinstance(m, torchvision.models.resnet.Bottleneck)):
        f, a, x = flops_Bottleneck(m, x)
    elif (isinstance(m, nn.AdaptiveAvgPool2d)):
        f, a, x = flops_AdaptiveAvgPool2d(m, x)
    elif (isinstance(m, nn.Linear)):
        f, a, x = flops_Linear(m, x)
    elif (isinstance(m, nn.Softmax)):
        f, a, x = flops_Softmax(m, x)
    elif (isinstance(m, SAM)):
        f, a, x = flops_SAM(m, x)
    elif (isinstance(m, Bottleneck)):
        f, a, x = flops_Bottleneck2(m, x)
    else:
        print("NOT FOUND:")
        print(m)
        exit(0)
    flops += f
    flops_add += a
    return flops, flops_add, x

def flops_Sequential_(model_list, x):
    flops = 0
    flops_add = 0
    for m in model_list:
        flops, flops_add, x = flops_any_layer(m, x, flops, flops_add)
    return flops, flops_add, x

def flops_resnet(model, x):
    flops = 0
    flops_add = 0
    # x = self.conv1(x)
    # x = self.bn1(x)
    # x = self.relu(x)
    # x = self.maxpool(x)
    # x = self.layer1(x)
    # x = self.layer2(x)
    # x = self.layer3(x)
    # x = self.layer4(x)
    # x = self.avgpool(x)
    # x = jt.reshape(x, (x.shape[0], -1))
    # x = self.fc(x)

    flops, flops_add, x = flops_Sequential_([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool], x)

    x = x.view(x.shape[0], -1)
    flops, flops_add, x = flops_any_layer(model.fc, x, flops, flops_add)
    return flops, flops_add, x

def flops_SAN(m, x):
    # x = self.relu(self.bn_in(self.conv_in(x)))
    # x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
    # x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
    # x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
    # x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
    # x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))
    # x = self.avgpool(x)
    flops, flops_add, x = flops_Sequential_(
        [
        m.conv_in, m.bn_in, m.relu,
        m.pool, m.conv0, m.layer0, m.bn0, m.relu,
        m.pool, m.conv1, m.layer1, m.bn1, m.relu,
        m.pool, m.conv2, m.layer2, m.bn2, m.relu,
        m.pool, m.conv3, m.layer3, m.bn3, m.relu,
        m.pool, m.conv4, m.layer4, m.bn4, m.relu,
        m.avgpool
        ], x)

    x = x.view(x.size(0), -1)
    flops, flops_add, x = flops_any_layer(m.fc, x, flops, flops_add)
    return flops, flops_add, x


def get_flops(model, input_size):
    x = torch.rand(input_size).cuda()
    model = model.cuda()
    if (isinstance(model, SAN)):
        return flops_SAN(model, x)
    elif (isinstance(model, torchvision.models.ResNet)):
        return flops_resnet(model, x)

def print_num(n):
    if (n < 1000):
        return str(n)
    elif (n < 1000000):
        return str(n / 1000) + 'K'
    elif (n < 1000000000):
        return str(n / 1000 / 1000) + 'M'
    else:
        return str(n / 1000 / 1000 / 1000) + 'G'

def get_param_size(model):
    size = 0
    size_Q = 0
    size_K = 0
    size_V = 0
    size_v_bn = 0
    size_tcn_conv = 0
    size_conv_pos2 = 0
    for name, p in model.named_parameters():
        s = 1
        for ss in p.shape:
            s *= ss
        size += s
        if ("conv2.Q" in name):
            size_Q += s
        if ("conv2.K" in name):
            size_K += s
        if ("conv2.V" in name):
            size_V += s
        if ("conv2.v_bn" in name):
            size_v_bn += s
        if ("conv2.tcn_conv" in name):
            size_tcn_conv += s
        if ("conv2.conv_pos2" in name):
            size_conv_pos2 += s
    print("Q:", print_num(size_Q))
    print("K:", print_num(size_K))
    print("V:", print_num(size_V))
    print("v_bn:", print_num(size_v_bn))
    print("tcn_conv:", print_num(size_tcn_conv))
    print("conv_pos2:", print_num(size_conv_pos2))
    return size

def analysis_model(model, size):
    print("Params:", print_num(get_param_size(model)))
    f, a, _ = get_flops(model, size)
    print("Flops:", print_num(f), print_num(a))

if __name__ == "__main__":
    jt.flags.use_cuda = 1
    print("=========resnet50============")
    model = jtmodels.__dict__['resnet50']()
    analysis_model(model, [1, 3, 224, 224])
    print("=====================")
    model = TNet.Resnet50()
    analysis_model(model, [1, 3, 224, 224])
    print('')

    print("=========resnet38============")
    model = jtmodels.__dict__['resnet38']()
    analysis_model(model, [1, 3, 224, 224])
    print("=====================")
    model = TNet.Resnet38()
    analysis_model(model, [1, 3, 224, 224])
    print('')

    print("=========resnet26============")
    model = jtmodels.__dict__['resnet26']()
    analysis_model(model, [1, 3, 224, 224])
    print("=====================")
    model = TNet.Resnet26()
    analysis_model(model, [1, 3, 224, 224])
    print('')
