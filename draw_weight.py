import global_args
global_args._init()
import jittor as jt
import numpy as np
from args import Args
import tcn_network as TNet
from tcn import TCN
import time
import jittor.models as jtmodels
from jittor import nn
import jittor.transform as transform
from jittor.dataset import ImageFolder
import logging
from tensorboardX import SummaryWriter
import os
from util.util import AverageMeter, smooth_loss, cal_accuracy, pytorch_conv_init
import shutil
from jittor import lr_scheduler
from san import san
from san_tcn import san_tcn
import cv2

jt.flags.use_cuda=1
jt.flags.compile_options={"max_parallel_depth":6}
args = Args().parse()
global_args.set_args(args)
print(args)

def main():
    best_acc1 = 0
    #TODO multi gpu
    if args.model == 'TNet26':
        model = TNet.Resnet26()
    elif args.model == 'TNet38':
        model = TNet.Resnet38()
    elif args.model == 'TNet50':
        model = TNet.Resnet50()
    elif args.model == 'Resnet26':
        model = jtmodels.__dict__['resnet26']()
    elif args.model == 'Resnet38':
        model = jtmodels.__dict__['resnet38']()
    elif args.model == 'Resnet50':
        model = jtmodels.__dict__['resnet50']()
    elif args.model == 'SAN10':
        model = san(sa_type = 0, layers=[2, 1, 2, 4, 1], kernels=[3, 7, 7, 7, 7], num_classes = 1000)
    elif args.model == 'SAN_TCN10':
        model = san_tcn(sa_type = 0, layers=[2, 1, 2, 4, 1], kernels=[3, 7, 7, 7, 7], num_classes = 1000)
    else:
        print("Model not found!")
        exit(0)
    if (args.use_pytorch_conv_init):
        pytorch_conv_init(model)

    model_path = os.path.join(args.save_path, 'model_best.pk')
    model.load(model_path)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # val_transform = transform.Compose([transform.Resize(256), transform.CenterCrop(224), transform.ImageNormalize(mean, std)])
    # [transform diff from val!vvvv]
    val_transform = transform.Compose([transform.Resize(224), transform.ImageNormalize(mean, std)])
    val_loader = ImageFolder('input_images', val_transform).set_attrs(batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers)
    test(val_loader, model)

def write_img(img, path):
    img = img.permute([1,2,0]) * 255
    img = jt.concat([img[:, :, 2:3], img[:, :, 1:2], img[:, :, 0:1]], 2)
    cv2.imwrite(path, img.data)

def write_imgs(imgss, path, pad=1):
    hn = len(imgss)
    wn = len(imgss[0])
    ch, h, w = imgss[0][0].shape
    h_ = (h + pad) * hn
    w_ = (w + pad) * wn
    out = []
    vertical = jt.ones([3, h, pad])
    horizontal = jt.ones([3, pad, (w + pad) + (wn - 1) * (imgss[0][1].shape[2] + pad)])
    # horizontal[0] = 1
    for i in range(hn):
        out_ = []
        for j in range(wn):
            out_.append(imgss[i][j])
            out_.append(vertical)
            # out[:, i * (h + pad) : i * (h + pad) + h, j * (w + pad) : j * (w + pad) + w] = imgss[i][j]
        out.append(jt.concat(out_, 2))
        out.append(horizontal)
    out = jt.concat(out, 1)
    write_img(out, path)

def draw_mask(img, ww):
    ch, h, w = img.shape
    fm_h, fm_w, h_, w_ = ww.shape
    center_fm_h = int(fm_h / 2)
    center_fm_w = int(fm_w / 2)
    # center_fm_h = 0
    # center_fm_w = 0
    h__ = int(h / fm_h)
    w__ = int(w / fm_w)
    mask = jt.zeros([ch, h, w])
    for i_ in range(h_):
        for j_ in range(w_):
            i = center_fm_h + i_ - int(h_ / 2)
            j = center_fm_w + j_ - int(w_ / 2)
            if (i < 0 or j < 0 or i >= fm_h or j >= fm_w):
                continue
            mask[:, i * h__:(i+1) * h__, j * w__:(j+1) * w__] = ww[center_fm_h, center_fm_w, i_, j_]
    mask_img = img / 2 + mask * 2
    # out = mask
    out = jt.concat([mask, mask_img], 2)
    return out

def draw(imgs, ws, path):
    bs, ch, h, w = imgs.shape
    _, fm_h, fm_w, ks, h_, w_ = ws.shape
    print(ws.shape)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    std_ = jt.array(std).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    mean_ = jt.array(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    imgs = imgs * std_ + mean_
    outs = []
    for i in range(bs):
        out = [imgs[i]]
        for j in range(ks):
            out.append(draw_mask(imgs[i], ws[i, :, :, j]))
        outs.append(out)
    write_imgs(outs, path)


def test(val_loader, model):
    model.eval()
    print(model)
    for i, (input, target) in enumerate(val_loader):
        output = model(input)
        w4 = model.layer4[0].conv2.weight
        w4 = w4.reshape(w4.shape[0], w4.shape[1], w4.shape[2], w4.shape[3], 7, 7)
        w3 = model.layer3[0].conv2.weight
        w3 = w3.reshape(w3.shape[0], w3.shape[1], w3.shape[2], w3.shape[3], 7, 7)
        w2 = model.layer2[1].conv2.weight
        w2 = w2.reshape(w2.shape[0], w2.shape[1], w2.shape[2], w2.shape[3], 7, 7)
        draw(input, w2, 'output_images/out.jpg')

if __name__ == '__main__':
    main()