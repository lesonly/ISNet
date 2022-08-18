#!/usr/bin/python3
#coding=utf-8


import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data4 import dataset
from net import EAFNet
import logging as logger
from lib4.data_prefetcher import DataPrefetcher
import numpy as np
import matplotlib.pyplot as plt
import pytorch_iou
from torch.utils.data.distributed import DistributedSampler
import argparse

TAG = "ours"
SAVE_PATH = "ours"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum



def EDA_loss(pred,mask, di,er):
    weit1  = 1+5*torch.abs(F.avg_pool2d(di, kernel_size=31, stride=1, padding=15)-di)
    weit2  = 1+5*torch.abs(F.avg_pool2d(er, kernel_size=31, stride=1, padding=15)-er)
    weight=weit1+weit2
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weight*wbce).sum(dim=(2,3))/weight.sum(dim=(2,3))
    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weight).sum(dim=(2,3))
    union = ((pred+mask)*weight).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

BASE_LR = 1e-3
MAX_LR = 0.1

# ------- 1. define loss function --------


iou_loss = pytorch_iou.IOU(size_average=True)

def train(Dataset, Network):
    
    parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
    parser.add_argument('--rank', default=0,
                        help='rank of current process')
    parser.add_argument('--word_size', default=4,
                        help="word size")
    parser.add_argument('--init_method', default='tcp://138.138.0.126:45678',
                        help="init-method")
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--batch_size','-b', default=8 ,type=int,help='batch-size')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg    = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train', batch=args.batch_size, lr=0.05, momen=0.9, decay=5e-4, epoch=36)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, num_workers=4, pin_memory=True, drop_last=True,sampler=DistributedSampler(data,shuffle=True))
    prefetcher = DataPrefetcher(loader)

    ## network
    net    = Network(cfg)
    net.train(True)
    net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)      

   
    ## parameter
    base, edger,head = [], [],[]
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        elif 'edge_net' in name:
            edger.append(param) #, {'params':edger}
        else:
            head.append(param)
    optimizer   = torch.optim.SGD([{'params':base},{'params':edger},{'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    db_size = len(loader)

    #training

    global_step = 0
    for epoch in range(cfg.epoch):
        loader.sampler.set_epoch(epoch)
        #scheduler.step()

        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask,enlarge_b,di,er= prefetcher.next()

        while image is not None:
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr #for backbone
            optimizer.param_groups[1]['lr'] = lr 
            optimizer.param_groups[2]['lr'] = lr #edger
            optimizer.momentum = momentum
        
            batch_idx += 1
            global_step += 1
                                   
            out2, out3, out4, out5,out2_coarse_a= net(image)  
            loss2  = EDA_loss(out2,mask, di,er)
            loss3  = EDA_loss(out3,mask, di,er)
            loss4  = EDA_loss(out4,mask, di,er)
            loss5  = EDA_loss(out5,mask, di,er)          
            loss_fine   = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4                 
            dilated=enlarge_b + mask
            dilated[dilated>1]=1
            loss_coarse=(F.binary_cross_entropy_with_logits(out2_coarse_a, dilated)+iou_loss(F.sigmoid(out2_coarse_a), dilated)).mean()
            loss=loss_fine+loss_coarse#+loss_hm

            optimizer.zero_grad()
            loss.backward()                          
            optimizer.step()
            
        
            if batch_idx % 10 == 0:               
                msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f | loss_fine=%.6f | loss_coarse=%.6f| loss=%.6f'% (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                    loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(),loss_fine.item(),loss_coarse.item(),loss.item())
                print(msg)
                logger.info(msg)
            
            image, mask,enlarge_b,di,er= prefetcher.next()
            
        if epoch == 36:
            torch.save(net.module.state_dict(), cfg.savepath+'/model'+str(epoch+1))



if __name__=='__main__':
    
    train(dataset, EAFNet)
