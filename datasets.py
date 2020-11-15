import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from devkit_semantics.devkit.helpers.labels import labels

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
TRAIN_DIR = "./data_semantics/training/image_2"
LABEL_DIR = "./data_semantics/training/semantic"
TRAIN_IMG = os.listdir(TRAIN_DIR)
LABEL_IMG = os.listdir(LABEL_DIR)
IMG_NAME = [i for i in TRAIN_IMG if i in LABEL_IMG]
TRAIN_NAME = IMG_NAME[:int(len(IMG_NAME) * TRAIN_RATIO)]
VALID_NAME = IMG_NAME[int(len(IMG_NAME) * TRAIN_RATIO):int(len(IMG_NAME) * (TRAIN_RATIO+VALID_RATIO))]
TEST_NAME = IMG_NAME[int(len(IMG_NAME) * (TRAIN_RATIO+VALID_RATIO)):]
CAT = 35
IMG_LENGTH = 352
IMG_HEIGHT = 1216


class KITTITRAIN(Dataset):
    def __init__(self):
        self.train_image_names = [os.path.join(TRAIN_DIR, img_name) for img_name in TRAIN_NAME]
        self.label_image_names = [os.path.join(LABEL_DIR, img_name) for img_name in TRAIN_NAME]

        self.input_transform = transforms.Compose([
            transforms.CenterCrop([IMG_LENGTH, IMG_HEIGHT]),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    def transform_label(self, ori):
        label = np.array(ori)
        label = label[(label.shape[0] - IMG_LENGTH) // 2:(label.shape[0] - IMG_LENGTH) // 2 + IMG_LENGTH,
                (label.shape[1] - IMG_HEIGHT) // 2:(label.shape[1] - IMG_HEIGHT) // 2 + IMG_HEIGHT]
        return torch.tensor(label).long()

    def __getitem__(self, index):
        img = Image.open(self.train_image_names[index]).convert('RGB')
        target = Image.open(self.label_image_names[index])
        img = self.input_transform(img)
        target = self.transform_label(target)
        return img, target

    def __len__(self):
        return len(self.train_image_names)


class KITTIVALID(Dataset):
    def __init__(self):
        self.train_image_names = [os.path.join(TRAIN_DIR, img_name) for img_name in TEST_NAME]
        self.label_image_names = [os.path.join(LABEL_DIR, img_name) for img_name in TEST_NAME]

        self.input_transform = transforms.Compose([
            transforms.CenterCrop([IMG_LENGTH, IMG_HEIGHT]),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    def transform_label(self, ori):
        label = np.array(ori)
        label = label[(label.shape[0] - IMG_LENGTH) // 2:(label.shape[0] - IMG_LENGTH) // 2 + IMG_LENGTH,
                (label.shape[1] - IMG_HEIGHT) // 2:(label.shape[1] - IMG_HEIGHT) // 2 + IMG_HEIGHT]
        return torch.tensor(label).long()

    def __getitem__(self, index):
        img = Image.open(self.train_image_names[index]).convert('RGB')
        target = Image.open(self.label_image_names[index])
        img = self.input_transform(img)
        target = self.transform_label(target)
        return img, target

    def __len__(self):
        return len(self.train_image_names)


class KITTITEST(Dataset):
    def __init__(self):
        self.train_image_names = [os.path.join(TRAIN_DIR, img_name) for img_name in VALID_NAME]
        self.label_image_names = [os.path.join(LABEL_DIR, img_name) for img_name in VALID_NAME]

        self.input_transform = transforms.Compose([
            transforms.CenterCrop([IMG_LENGTH, IMG_HEIGHT]),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    def transform_label(self, ori):
        label = np.array(ori)
        label = label[(label.shape[0] - IMG_LENGTH) // 2:(label.shape[0] - IMG_LENGTH) // 2 + IMG_LENGTH,
                (label.shape[1] - IMG_HEIGHT) // 2:(label.shape[1] - IMG_HEIGHT) // 2 + IMG_HEIGHT]
        return torch.tensor(label).long()

    def __getitem__(self, index):
        img = Image.open(self.train_image_names[index]).convert('RGB')
        target = Image.open(self.label_image_names[index])
        transformed_img = self.input_transform(img)
        target = self.transform_label(target)
        return transformed_img, target, self.train_image_names[index]

    def __len__(self):
        return len(self.train_image_names)
