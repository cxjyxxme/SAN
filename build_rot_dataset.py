import cv2
import os
import random
import json as js
in_s = 224
rotate_range = 45
resize_rate_range = [0.5, 2]
fm_sizes = [7,14,28,56]

def get_img2(img, c):
    h, w, _ = img.shape
    rotation = (random.random() * 2 - 1) * rotate_range
    scale = random.random() * (resize_rate_range[1] - resize_rate_range[0]) + resize_rate_range[0]
    M = cv2.getRotationMatrix2D((c[1], c[0]), rotation, scale)
    #进行仿射变换，第一个参数图像，第二个参数是旋转矩阵，第三个参数是变换之后的图像大小
    img_out = cv2.warpAffine(img, M, (w, h))
    return img_out, rotation, scale

cnt = 0
output_dir = "rot_dataset"
datas = []
tot = 0
for root, dirs, files in os.walk("dataset/ILSVRC2012/val"):
    for name in files:
        if (not name.endswith('.JPEG')):
            continue
        tot += 1
        print(tot, '/ 50000')
        path = os.path.join(root, name)
        img = cv2.imread(path)
        h, w, _ = img.shape
        img1 = cv2.resize(img, (in_s,in_s))
        cv2.imwrite(os.path.join(output_dir, name), img1)
        cnt = 0
        for s in fm_sizes:
            cnt += 1
            low = int(round(s / 4))
            high = s - low - 1
            x = random.randint(low, high) #H
            y = random.randint(low, high) #W
            c_x = h // s // 2 * (1 + x * 2)
            c_y = w // s // 2 * (1 + y * 2)
            # img_ = cv2.circle(img, (c_y, c_x), 5, (255,0,0))
            img2, rotation, scale = get_img2(img, (c_x, c_y))
            img2 = cv2.resize(img2, (in_s, in_s))
            cv2.imwrite(os.path.join(output_dir, name[:-5] + '_'+str(cnt)+'.JPEG'), img2)
            data = {
                'img1_path':os.path.join(output_dir, name),
                'img2_path':os.path.join(output_dir, name[:-5] + '_'+str(cnt)+'.JPEG'),
                'fm_size':s,
                'x':x,
                'y':y,
                'rotation':rotation,
                'scale':scale
            }
            datas.append(data)
js.dump(datas, open(os.path.join(output_dir, 'data.json'), "w"))
    