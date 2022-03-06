#!/usr/bin/python3
# coding=utf-8

import cv2
import torch
import numpy as np
import random
import torchvision.transforms as transforms


class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask, edge,region,region2):
        for op in self.ops:
            image, mask, edge,region,region2 = op(image, mask, edge,region,region2)
        return image, mask, edge,region,region2


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, edge,region,region2):
        image = (image - self.mean) / self.std
        mask /= 255
        edge /= 255
        region /= 255
        region2 /=255
        return image, mask, edge,region,region2


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, edge,region,region2):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        region = cv2.resize(region, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        region2 = cv2.resize(region2, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, edge,region,region2


class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, edge,region,region2):
        H, W, _ = image.shape
        xmin = np.random.randint(W - self.W + 1)
        ymin = np.random.randint(H - self.H + 1)
        image = image[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask = mask[ymin:ymin + self.H, xmin:xmin + self.W, :]
        edge = edge[ymin:ymin + self.H, xmin:xmin + self.W, :]
        region= region[ymin:ymin + self.H, xmin:xmin + self.W, :]
        region2= region2[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return image, mask, edge,region,region2


class RandomHorizontalFlip(object):
    def __call__(self, image, mask, edge,region,region2):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
            edge = edge[:, ::-1, :].copy()
            region = region[:, ::-1, :].copy()
            region2 = region2[:, ::-1, :].copy()
        return image, mask, edge,region,region2


class ToTensor(object):
    def __call__(self, image, mask, edge,region,region2):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        edge = torch.from_numpy(edge)
        edge = edge.permute(2, 0, 1)
        region = torch.from_numpy(region)
        region = region.permute(2, 0, 1)
        region2 = torch.from_numpy(region2)
        region2 = region2.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True), edge.mean(dim=0, keepdim=True),region.mean(dim=0, keepdim=True),region2.mean(dim=0, keepdim=True)

class Hide_patch(object):

    def __call__(self, img, mask, edge,region,region2):
        wd, ht, _ = img.shape

        # possible grid size, 0 means no hiding
        grid_sizes=[0,16,32,44,56]

        # hiding probability
        hide_prob = 0.2
    
        # randomly choose one grid size
        grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

        # hide the patches
        if(grid_size!=0):
            for x in range(0,wd,grid_size):
                for y in range(0,ht,grid_size):
                    x_end = min(wd, x+grid_size)  
                    y_end = min(ht, y+grid_size)
                    if(random.random() <=  hide_prob):
                        img[x:x_end,y:y_end,:]=0

        return img, mask, edge,region,region2
