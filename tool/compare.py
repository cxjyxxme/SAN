import global_args
global_args._init()
import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist

from model.san import san
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, cal_accuracy
from skimage import io
import json as js
import math

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
data_path = 'rot_dataset'
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/imagenet/imagenet_san10_pairwise.yaml', help='config file')
    parser.add_argument('opts', help='see config/imagenet/imagenet_san10_pairwise.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

class SAM_(nn.Module):
    def __init__(self, sa_type, layers, kernels, classes):
        super(SAM_, self).__init__()
        self.module = san(sa_type, layers, kernels, classes)
    
    def forward(self, x):
        return self.module(x)

def get_score(data, ws):
    ss = 56
    cc = (ss - 1) / 2
    w1s = []
    w2s = []
    for w in ws:
        w1s.append(w[0][data['x']][data['y']])
        w2s.append(w[1][data['x']][data['y']])
    w1 = torch.cat(w1s, 0).permute([1,2,0]).detach().cpu().numpy()
    w1 = cv2.resize(w1, (ss, ss), interpolation=cv2.INTER_NEAREST)

    w2 = torch.cat(w2s, 0).permute([1,2,0]).detach().cpu().numpy()
    w2 = cv2.resize(w2, (ss, ss), interpolation=cv2.INTER_NEAREST)

    M = cv2.getRotationMatrix2D((cc, cc), data['rotation'], data['scale'])
    w1_ = cv2.warpAffine(w1, M, (ss, ss))
    mask = np.ones([ss,ss,1])
    mask_ = cv2.warpAffine(mask, M, (ss, ss)).reshape([ss, ss, 1])
    w2_ = w2
    # M = cv2.getRotationMatrix2D((cc, cc), 0, 1)
    # w2_ = cv2.warpAffine(w2, M, (ss, ss))
    err = (np.abs(w1_ - w2_) * mask_).sum() / mask_.sum() / w1_.shape[2]
    return err
    
def get_distribution(data, distribution, w):
    p = w.shape[-1]
    c = p // 2
    w = w[:, data['x'], [data['y']]]
    w = w.reshape(-1, p, p)
    for i in range(c + 1):
        ss = w[:, c - i : c + i + 1, c - i : c + i + 1]
        ss = ss.sum(1).sum(1)
        if (i > 0):
            ss -= last_ss
        last_ss = ss
        distribution[i].append(ss.mean().item())
    
def get_distribution2(distribution, w):
    p = w.shape[-1]
    c = p // 2
    w = w.reshape(-1, p, p)
    for i in range(c + 1):
        ss = w[:, c - i : c + i + 1, c - i : c + i + 1]
        ss = ss.sum(1).sum(1)
        if (i > 0):
            ss -= last_ss
        last_ss = ss
        distribution[i].append(ss.mean().item())

def print_distribution(d):
    for i in range(10):
        print("=========",i,"===========")
        for j in range(4):
            sum = 0
            cnt = 0
            for k in d[i][j]:
               sum += k
               cnt += 1
            if (cnt == 0):
                print('0', end=' ')
            else:
                print(sum / cnt, end=' ')
        print("")

def main():
    global args, logger
    args = get_parser()
    global_args.set_args(args)
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    model = SAM_(args.sa_type, args.layers, args.kernels, args.classes).cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    if os.path.isdir(args.save_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    datas = js.load(open(os.path.join(data_path, 'data.json'), "r"))
    model.eval()
    scores = []
    sum = 0
    distribution = []
    distribution2 = []
    for i in range(10):
        distribution.append([])
        distribution2.append([])
        for j in range(4):
            distribution[i].append([])
            distribution2[i].append([])
    random.shuffle(datas)
    test_distribution = True
    for i in range(len(datas)):
        print(i, '/', len(datas))
        data = datas[i]

        img1 = io.imread(data['img1_path'])
        img1 = val_transform(img1)
        img2 = io.imread(data['img2_path'])
        img2 = val_transform(img2)
        
        input = torch.stack([img1, img2], 0).cuda(non_blocking=True)
        _ = model(input)
        ws = []
        cnt_ = 0
        for layer in [model.module.layer0, model.module.layer1, model.module.layer2, model.module.layer3, model.module.layer4]:
            for k in range(len(layer)):
                w = layer[k].sam.weight
                w = w.permute([0,3,1,2])
                bs, hw, ks, pp = w.shape
                w = w.reshape([bs, int(math.sqrt(hw)), int(math.sqrt(hw)), ks, int(math.sqrt(pp)), int(math.sqrt(pp))])

                if (w.shape[1] == data['fm_size']):
                    ws.append(w)
                    if (test_distribution):
                        get_distribution(data, distribution[cnt_], w)
                if (test_distribution):
                    get_distribution2(distribution2[cnt_], w)
                cnt_ += 1
        if (not test_distribution):
            s = get_score(data, ws)
            sum += s
            scores.append(s)
            print(sum / (i + 1))
        if (i > 300):
            break
    if (test_distribution):
        print_distribution(distribution)
        print_distribution(distribution2)
    else:
        sum_score = 0
        for s in scores:
            sum_score += s
        print(sum_score / len(scores))

#     val_set = torchvision.datasets.ImageFolder('input_images', val_transform)
#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)
#     validate(val_loader, model)

# def write_img(img, path):
#     img = img.permute([1,2,0]) * 255
#     img = torch.cat([img[:, :, 2:3], img[:, :, 1:2], img[:, :, 0:1]], 2)
#     cv2.imwrite(path, img.detach().cpu().numpy())

# def write_imgs(imgss, path, pad=1):
#     hn = len(imgss)
#     wn = len(imgss[0])
#     ch, h, w = imgss[0][0].shape
#     h_ = (h + pad) * hn
#     w_ = (w + pad) * wn
#     out = []
#     vertical = torch.ones([3, h, pad]).cuda()
#     horizontal = torch.ones([3, pad, (w + pad) + (wn - 1) * (imgss[0][1].shape[2] + pad)]).cuda()
#     # horizontal[0] = 1
#     for i in range(hn):
#         out_ = []
#         for j in range(wn):
#             out_.append(imgss[i][j])
#             out_.append(vertical)
#             # out[:, i * (h + pad) : i * (h + pad) + h, j * (w + pad) : j * (w + pad) + w] = imgss[i][j]
#         out.append(torch.cat(out_, 2))
#         out.append(horizontal)
#     out = torch.cat(out, 1)
#     write_img(out, path)

# def draw_mask(img, ww):
#     ch, h, w = img.shape
#     fm_h, fm_w, h_, w_ = ww.shape
#     center_fm_h = int(fm_h / 2)
#     center_fm_w = int(fm_w / 2)
#     # center_fm_h = 0
#     # center_fm_w = 0
#     h__ = int(h / fm_h)
#     w__ = int(w / fm_w)
#     mask = torch.zeros([ch, h, w]).cuda()
#     for i_ in range(h_):
#         for j_ in range(w_):
#             i = center_fm_h + i_ - int(h_ / 2)
#             j = center_fm_w + j_ - int(w_ / 2)
#             if (i < 0 or j < 0 or i >= fm_h or j >= fm_w):
#                 continue
#             mask[:, i * h__:(i+1) * h__, j * w__:(j+1) * w__] = ww[center_fm_h, center_fm_w, i_, j_]
#     mask_img = img / 2 + mask * 2
#     # out = mask
#     out = torch.cat([mask*3, mask_img], 2)
#     return out

# def draw(imgs, ws, path):
#     bs, ch, h, w = imgs.shape
#     _, fm_h, fm_w, ks, h_, w_ = ws.shape
#     mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#     std_ = torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
#     mean_ = torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
#     imgs = imgs * std_ + mean_
#     outs = []
#     for i in range(bs):
#         out = [imgs[i]]
#         for j in range(ks):
#             out.append(draw_mask(imgs[i], ws[i, :, :, j]))
#         outs.append(out)
#     write_imgs(outs, path)

# import math
# def validate(val_loader, model):
#     id = args.save_path.split('/')[-2][len("san10_pairwise"):]
#     model.eval()
#     for i, (input, target) in enumerate(val_loader):
#         input = input.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)

#         with torch.no_grad():
#             output = model(input)
#             #[bs, share, pp, h*w]
#             w = model.module.layer4[0].sam.weight
#             # w = model.module.layer3[3].sam.weight
#             w = model.module.layer3[0].sam.weight
            
#             w = w.permute([0,3,1,2])
#             bs, hw, ks, pp = w.shape
#             w = w.reshape([bs, int(math.sqrt(hw)), int(math.sqrt(hw)), ks, 7, 7])
#             draw(input, w, 'output_images/out' +str(id) +'.jpg')

if __name__ == '__main__':
    main()
