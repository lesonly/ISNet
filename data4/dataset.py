#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np

import torchvision.transforms as transforms
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset
#color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'ECSSD' in self.kwargs['datapath']:
            self.mean = np.array([[[117.15, 112.48, 92.86]]])
            self.std = np.array([[[56.36, 53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean = np.array([[[124.55, 118.90, 102.94]]])
            self.std = np.array([[[56.77, 55.97, 57.50]]])
        elif 'DUT-OMRON' in self.kwargs['datapath']:
            self.mean = np.array([[[120.61, 121.86, 114.92]]])
            self.std = np.array([[[58.10, 57.16, 61.09]]])
        elif 'MSRA-10K' in self.kwargs['datapath']:
            self.mean = np.array([[[115.57, 110.48, 100.00]]])
            self.std = np.array([[[57.55, 54.89, 55.30]]])
        elif 'MSRA-B' in self.kwargs['datapath']:
            self.mean = np.array([[[114.87, 110.47, 95.76]]])
            self.std = np.array([[[58.12, 55.30, 55.82]]])
        elif 'SED2' in self.kwargs['datapath']:
            self.mean = np.array([[[126.34, 133.87, 133.72]]])
            self.std = np.array([[[45.88, 45.59, 48.13]]])
        elif 'PASCAL-S' in self.kwargs['datapath']:
            self.mean = np.array([[[117.02, 112.75, 102.48]]])
            self.std = np.array([[[59.81, 58.96, 60.44]]])
        elif 'HKU-IS' in self.kwargs['datapath']:
            self.mean = np.array([[[123.58, 121.69, 104.22]]])
            self.std = np.array([[[55.40, 53.55, 55.19]]])
        elif 'SOD' in self.kwargs['datapath']:
            self.mean = np.array([[[109.91, 112.13, 93.90]]])
            self.std = np.array([[[53.29, 50.45, 48.06]]])
        elif 'THUR15K' in self.kwargs['datapath']:
            self.mean = np.array([[[122.60, 120.28, 104.46]]])
            self.std = np.array([[[55.99, 55.39, 56.97]]])
        elif 'SOC' in self.kwargs['datapath']:
            self.mean = np.array([[[120.48, 111.78, 101.27]]])
            self.std = np.array([[[58.51, 56.73, 56.38]]])
        else:
            # raise ValueError
            self.mean = np.array([[[0.485 * 256, 0.456 * 256, 0.406 * 256]]])
            self.std = np.array([[[0.229 * 256, 0.224 * 256, 0.225 * 256]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        with open(os.path.join(cfg.datapath, cfg.mode + '.txt'), 'r') as lines:
            self.samples = []
            for line in lines:
                imagepath = os.path.join(cfg.datapath, 'image', line.strip() + '.jpg')
                maskpath = os.path.join(cfg.datapath, 'mask', line.strip() + '.png')
                edgepath = os.path.join(cfg.datapath, 'mask_enlarged_b', line.strip() + '.png')
                regionpath = os.path.join(cfg.datapath, 'mask_d', line.strip() + '.png')
                regionpath2 = os.path.join(cfg.datapath, 'mask_e', line.strip() + '.png')
                self.samples.append([imagepath, maskpath, edgepath,regionpath,regionpath2])

        if cfg.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(320, 320),
                                               transform.RandomHorizontalFlip(),
                                             
                                               transform.Hide_patch(),
                                               transform.RandomCrop(288, 288),                                              
                                               transform.ToTensor())
            
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(320, 320),
                                               transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath, edgepath,regionpath,regionpath2 = self.samples[idx]
        image = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
        mask = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
        edge = cv2.imread(edgepath).astype(np.float32)[:, :, ::-1]
        region = cv2.imread(regionpath).astype(np.float32)[:, :, ::-1]
        region2 = cv2.imread(regionpath2).astype(np.float32)[:, :, ::-1]
        H, W, C = mask.shape
        image, mask, edge,region,region2 = self.transform(image, mask, edge,region,region2)
        return image, mask, edge,region,region2, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)

