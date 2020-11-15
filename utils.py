# coding=utf-8
# Created on 2020-11-13 16:15
# Copyright © 2020 Alan. All rights reserved.
import torch
import numpy as np
from datasets import IMG_LENGTH, IMG_HEIGHT
from sklearn.metrics import confusion_matrix

SMOOTH = 1e-6
ALL_PEXEL = IMG_LENGTH * IMG_HEIGHT


def get_meaniou(outputs, labels):
    outputs, labels = outputs.cpu().numpy(), labels.cpu().numpy()
    outputs = outputs.squeeze(0).flatten()
    labels = labels.squeeze(0).flatten()
    cm = np.array(confusion_matrix(outputs, labels))
    t = cm.diagonal()
    all_true = sum(t)
    curr_iou = 0.0
    for tp in t:
        tn = all_true - tp

        curr_iou += (tp + SMOOTH) / (ALL_PEXEL - tn + SMOOTH)
    return curr_iou / t.shape[0]

