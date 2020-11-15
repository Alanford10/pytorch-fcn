from torchvision.models import vgg16
from torchvision.models._utils import IntermediateLayerGetter
from torch.optim import SGD
import torch.nn as nn
import time
from datasets import *
from utils import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device using:', device)


class FCN32(nn.Module):
    def __init__(self):
        super(FCN32, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score = nn.Conv2d(4096, CAT, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(CAT, CAT, kernel_size=(32, 32), stride=32, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.score(x))
        x = self.deconv(x)
        return x


class FCN16(nn.Module):
    def __init__(self):
        super(FCN16, self).__init__()
        features = vgg16(pretrained=True).features
        self.get_layer = IntermediateLayerGetter(features, {'23': 'modified_23', '30': 'modified_30'})
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score = nn.Conv2d(4096, CAT, kernel_size=1)
        self.side_conv = nn.Conv2d(512, CAT, kernel_size=1)
        self.deconv1 = nn.ConvTranspose2d(CAT, CAT, kernel_size=(2, 2), stride=2, padding=0, bias=False)
        self.deconv2 = nn.ConvTranspose2d(CAT, CAT, kernel_size=(16, 16), stride=16, padding=0, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return_dict = self.get_layer(x)
        y, x = return_dict['modified_23'], return_dict['modified_30']
        y = self.side_conv(y)
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.score(x))
        x = self.relu(self.deconv1(x))
        x += y
        x = self.deconv2(x)
        return x


def train(model_name, epoch):
    if model_name == "FCN16":
        model = FCN16()
    else:
        model = FCN32()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.99)
    train_loader = torch.utils.data.DataLoader(KITTITRAIN(), batch_size=5, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(KITTIVALID(), batch_size=1, shuffle=True)
    train_loss, valid_loss, valid_iou = [], [], []
    # start_time = time.time()
    for epoch in range(1, epoch+1):
        # print("Training for No.{} epoch".format(epoch))
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # images = Variable(images)
            # labels = Variable(labels, requires_grad=False)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        t_loss = running_loss / train_loader.__len__()
        train_loss.append(t_loss)

        with torch.no_grad():
            print("Validate for No.{} item".format(epoch))
            v_loss = 0.0
            v_iou = 0.0
            for step, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                v_loss += criterion(outputs, labels).item()
                outputs = torch.argmax(outputs, dim=1)
                v_iou += get_meaniou(outputs, labels)
            v_loss /= valid_loader.__len__()
            v_iou /= valid_loader.__len__()
            valid_loss.append(v_loss)
            valid_iou.append(v_iou)
            # set early stopping to be 2 epoch
            if min(valid_loss) not in valid_loss[max(len(valid_loss)-2, 0):]:
                break

        torch.save(model, "{}_model.pth".format(model_name))
        print("Epoch {}: trainloss {} validloss {} validIOU {}".format(epoch, t_loss, v_loss, v_iou))
        # print("Time taken: {} hours".format((time.time() - start_time) / 3600.0))
    return model, train_loss, valid_loss, valid_iou

