import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from dataset import *
import argparse
import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from time import strftime, localtime
import math
from torchvision.models.vgg import model_urls

class MultipleOptimizer(object):
    def __init__(self,opt1,opt2):
        self.optimizer1 = opt1
        self.optimizer2 = opt2

    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        return

    def step(self,stage = 'stage1'):

        if stage == 'stage2':
            self.optimizer1.step()
        self.optimizer2.step()

        return

class BiLinearModel(nn.Module):
    def __init__(self):

        super(BiLinearModel, self).__init__()

        model_urls['vgg16'] = model_urls['vgg16'].replace('https://','http://')
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])
        self.fc1 = torch.nn.Sequential(
                    torch.nn.Linear(512**2,200),
                    torch.nn.PReLU())
        self.fc2 = torch.nn.Linear(200,2)
        # self.fc = torch.nn.Sequential(
        #             torch.nn.Linear(512**2, 200),
        #             torch.nn.PReLU(),
        #             torch.nn.Linear(200,2))



    def forward(self, x1,x2):

        batch_size = x1.size()[0]
        x1 = self.features(x1).view(batch_size,-1,14*14)
        x2 = self.features(x2).view(batch_size,-1,14*14)
        x = torch.bmm(x1,torch.transpose(x2,1,2)) / (14*14)
        assert x.size() == (batch_size,512,512)
        x = x.view(batch_size,-1)
        x = torch.sqrt(x+1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.fc2(self.fc1(x))
        assert x.size() == (batch_size,2)


        # batch_size = x.size()[0]
        # x1 = self.features(x)
        # x1 = x1.view(batch_size,-1,28*28)
        # x = torch.bmm(x1,torch.transpose(x1,1,2)) / (28*28)
        # assert x.size() == (batch_size,512,512)
        # x = x.view(batch_size,-1)
        # x = torch.sqrt(x+1e-5)
        # x = torch.nn.functional.normalize(x)
        # x = self.fc(x)
        # assert x.size() == (batch_size,2)
        return x

    def loadweight_from(self, pretrain_path):
        pretrained_dict = torch.load(pretrain_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def cls_loss(self, x, y):
        loss = F.nll_loss(F.log_softmax(x, dim=1), y)
        return loss
def train(model,trainloader,validloader,opt, lr = 0.001, num_epochs=200,train_log = 'logs/bilinear_basic.txt',model_name = 'models/bilinear_basic/'):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    fd = open(train_log, 'w+')
    for epoch in range(num_epochs):

        model.train()
        torch.set_grad_enabled(True)
        train_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (im1,im2, label) in enumerate(trainloader):

            im1,im2 = im1.float(),im2.float()
            if use_cuda:
                im1,im2, label = im1.cuda(), im2.cuda(), label.cuda()
            opt.zero_grad()
            im1, im2, label = Variable(im1), Variable(im2), Variable(label)
            feat = model.forward(im1, im2)
            loss = model.cls_loss(feat, label)
            loss.backward()
            if epoch < 2:
                opt.step('stage1')
            else:
                opt.step('stage2')
            train_loss += loss
            total += label.size(0)
            _, predicted = torch.max(feat.data, 1)
            correct += (predicted == label).sum().item()
            # print("    #Iter %3d: Training Loss: %.3f" % (
            #   batch_idx, loss.data[0]))
            # # print(total)
        train_acc = correct/float(total)


        # validate
        model.eval()
        torch.set_grad_enabled(False)
        valid_loss = 0.0
        total = 0
        correct = 0
        tp = 0
        tn = 0
        pos = 0
        neg = 0
        for batch_idx, (im1,im2, label) in enumerate(validloader):
            im1, im2 = im1.float(), im2.float()
            if use_cuda:
                im1, im2, label = im1.cuda(), im2.cuda(), label.cuda()
            im1, im2, label = Variable(im1), Variable(im2), Variable(label)
            feat = model.forward(im1,im2)
            valid_loss += model.cls_loss(feat, label)
            # compute the accuracy
            total += label.size(0)

            _, predicted = torch.max(feat.data, 1)
            pos += label.sum()
            neg += (1 - label).sum()
            correct += (predicted == label).sum().item()
            tp += (predicted * label).sum().item()
            tn += ((1 - predicted) * (1 - label)).sum().item()
        valid_acc = correct / float(total)
        tpr = float(tp) / float(pos)
        tnr = float(tn) / float(neg)
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        print("#Epoch {}: Train Loss: {:.4f},Train Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}, Valid tpr: {:.4f}, Valid tnr: {:.4f}".
               format(epoch, train_loss / len(trainloader.dataset),train_acc, valid_loss / len(validloader.dataset), valid_acc, tpr, tnr))
        fd.write('#Epoch {}: Train Loss: {:.4f},Train Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}, Valid tpr: {:.4f}, Valid tnr: {:.4f} \n'.
               format(epoch, train_loss / len(trainloader.dataset),train_acc, valid_loss / len(validloader.dataset), valid_acc, tpr, tnr))
        torch.save(model.state_dict(),model_name +'{}_epoch-{}_{:.4f}.pth'.format('bilinear_', epoch, valid_acc))



parser = argparse.ArgumentParser(description='Bilinear model')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--modelname', type=str, default='bilinear_')
parser.add_argument('--pretrained', type=bool, default=True)
args = parser.parse_args()
normal = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                                std = (0.29, 0.224, 0.225))
                            ])

train_transforms = [normal]
test_transforms = [normal]

train_loader = torch.utils.data.DataLoader(
    FineGrainVerificationDataset('/media/ramdisk/cars_data/first_100_from_train_balanced.txt',
                    transform= train_transforms),batch_size=16, shuffle=True, num_workers = 4)
valid_loader = torch.utils.data.DataLoader(
    FineGrainVerificationDataset('/media/ramdisk/cars_data/last_100_from_test_short.txt',
                    transform= test_transforms), batch_size=16, shuffle=False, num_workers = 4)

net = BiLinearModel()
net.features = torch.nn.DataParallel(net.features).cuda()
net.loadweight_from('/media/ramdisk/cars_data/f100_bilinear__0.8723_epoch-106.pth')
params1 = []
params2 = []
for name,x in net.named_parameters():
    if 'fc2' in name:
        params2.append(x)
    else:
        params1.append(x)


optimizer1 = torch.optim.Adam(params1,lr = args.lr,weight_decay = 1e-4)
optimizer2 = torch.optim.Adam(params2,lr = args.lr,weight_decay = 1e-4)
opt = MultipleOptimizer(optimizer1, optimizer2)


train(net,train_loader, valid_loader,opt,lr=args.lr, num_epochs=args.epochs, train_log = 'logs/exp1.txt', model_name = 'models/')
verification_test(net,verification_loader)