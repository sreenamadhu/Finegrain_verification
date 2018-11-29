from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import pandas as pd
from skimage import io
from skimage import color
import os
import os.path
import numpy as np
from torchvision import datasets, transforms
import cv2
from random import randint
import random
import torch
from random import randint
import random

class FineGrainVerificationDataset(data.Dataset):
    def __init__(self, list, transform=None, mirror=False):

        self.pairs = pd.read_csv(list, header = None)
        self.transform = transform
        self.mirror = mirror
        self.list = list

    def __getitem__(self, index):
        im1_path = self.pairs.ix[index,0]
        im2_path = self.pairs.ix[index,2]
        data_dir = '/media/ramdisk/cars_data/'
        # data_dir = ''        
        im1 = io.imread(data_dir + im1_path)
        im2 = io.imread(data_dir + im2_path)
        if len(im1.shape) < 3:
            im1 = np.repeat(im1[:, :, np.newaxis], 3, 2)
        if len(im2.shape) < 3:
            im2 = np.repeat(im2[:, :, np.newaxis], 3, 2)
        im1 = Image.fromarray(im1, mode='RGB')
        im2 = Image.fromarray(im2, mode='RGB')
        label = int(self.pairs.ix[index,1])
        

        ind1 = randint(0,5)
        ind2 = randint(0,5)

        if self.transform is not None:
            if len(self.transform) > 1:
                im1 = self.transform[ind1](im1)
                im2 = self.transform[ind2](im2)
            else:
                im1 = self.transform[0](im1)
                im2 = self.transform[0](im2)
        
        return im1,im2,label

    def __len__(self):
        return len(self.pairs)


class PairDataLoader(object):

    def __init__(self, dataset = None, batch_size = 16, shuffle = True):

        self.dataset = dataset
        self.num_imgs = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_gen()

    def __len__(self):
        return batch_size

    def __iter__(self):

        self.data_gen()
        return self

    def data_gen(self):

        self.flag = 'positive'
        self.pos_current = 0
        self.pos_prev = 0
        self.neg_current = 0
        self.neg_prev = 0
        labels = []
        with open(self.dataset.list) as f:
             for line in f:
                line = line.strip('\r\n').split(',')
                labels.append(int(line[1]))
        labels = np.array(labels)

        self.pos_indexes = np.where(labels==1)[0]
        self.neg_indexes = np.where(labels==0)[0]

        if self.shuffle:
            random.shuffle(self.pos_indexes)
            random.shuffle(self.neg_indexes)

    def collate_fn(self,data):
        # print(data)
        data.sort(key = lambda x: len(x[1]), reverse = True)
        images1,images2,captions = zip(*data)

        images1 = torch.stack(images1,0)
        images2 = torch.stack(images2,0)
        captions = torch.from_numpy(np.array(captions))
        return images1,images2,captions

    def next(self):

        if self.flag == 'positive':
            self.pos_current = self.pos_current + int(self.batch_size)
            self.indexes = list(self.pos_indexes[self.pos_prev : self.pos_current])
            random.shuffle(self.indexes)
            self.pos_prev = self.pos_prev + int(self.batch_size)
            self.flag = 'negative'

        else:
            self.neg_current = self.neg_current + int(self.batch_size)
            self.indexes = list(self.neg_indexes[self.neg_prev : self.neg_current])
            random.shuffle(self.indexes)
            self.neg_prev = self.neg_prev + int(self.batch_size)
            self.flag = 'positive'
        
        if self.pos_current > len(self.pos_indexes) and self.neg_current > len(self.neg_indexes):
            raise StopIteration
        else:
            return self.collate_fn([self.dataset[index] for index in self.indexes])

'''
----------------------Usage--------------------------------------------
train_dir = '/media/ramdisk/data/open_set_train_pairwise_balanced.txt'
test_dir = '/media/ramdisk/data/open_set_test_pairwise_balanced_short.txt'
train_loader = PairDataLoader(
    FineGrainVerificationDataset(train_dir, transform= vgg_transform),
    batch_size=64, shuffle=True)

test_loader = PairDataLoader(
    FineGrainVerificationDataset(test_dir, transform= vgg_transform),
    batch_size=64, shuffle=True)

'''
