# coding=utf-8
# Created on 2020-11-14 13:51
# Copyright © 2020 Alan. All rights reserved.
import torch
from datasets import KITTITEST, IMG_LENGTH, IMG_HEIGHT
from devkit_semantics.devkit.helpers.labels import labels
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ref = {}
for sub_label in labels:
    ref[sub_label[1]] = sub_label[7]


def decode_segmap(image):
    canvas = np.zeros((3, IMG_LENGTH, IMG_HEIGHT))
    for i in range(IMG_LENGTH):
        for j in range(IMG_HEIGHT):
            canvas[0, i, j] = ref[image[i][j]][0]
            canvas[1, i, j] = ref[image[i][j]][1]
            canvas[2, i, j] = ref[image[i][j]][2]
    return canvas


def center_crop(ori):
    label = np.array(ori)
    label = label[(label.shape[0]-IMG_LENGTH)//2:(label.shape[0]-IMG_LENGTH)//2+IMG_LENGTH,
            (label.shape[1]-IMG_HEIGHT)//2:(label.shape[1]-IMG_HEIGHT)//2+IMG_HEIGHT, :]
    return label


# new folder named for fcn16 and fcn32
if not os.path.exists('fcn16_res'):
    os.mkdir('fcn16_res')
if not os.path.exists('fcn32_res'):
    os.mkdir('fcn32_res')

model = torch.load("model_epoch18.pth", map_location=torch.device('cpu'))
test_loader = torch.utils.data.DataLoader(KITTITEST(), batch_size=1, shuffle=True)

with torch.no_grad():
    for step, (transformed_images, label, img_name) in enumerate(test_loader):
        outputs = model(transformed_images)
        outputs = torch.argmax(outputs, dim=1)
        img_name = img_name[0]
        outputs = outputs.numpy().squeeze(0)
        seg_img = decode_segmap(outputs)
        seg_img = seg_img.transpose((1, 2, 0)).astype(int)
        ori_image = center_crop(np.array(Image.open(img_name)))
        new_image = np.concatenate((ori_image, seg_img), axis=0)
        plt.imshow(new_image)
        plt.savefig(os.path.join('fcn16_res', img_name.split('/')[-1]))
