#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import dataset

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # 权值初始化，kaiming正态分布：此为0均值的正态分布，N～ (0,std)，其中std = sqrt(2/(1+a^2)*fan_in)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.ReLU6):
            pass
        else:
            m.initialize()


# resnet组件
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./resnet50-19c8e357.pth'), strict=False)


""" Channel Attention Module """
class CALayer(nn.Module):
    def __init__(self, in_ch_left, in_ch_down):
        super(CALayer, self).__init__()
        self.conv0 = nn.Conv2d(in_ch_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_ch_down, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256
        down = down.mean(dim=(2, 3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return left * down

    def initialize(self):
        weight_init(self)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    def initialize(self):
        weight_init(self)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    def initialize(self):
        weight_init(self)

class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=16):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return out_w * out_h*short

    def initialize(self):
        weight_init(self)
""" Body_Aggregation1 Module """
class BA1(nn.Module):
    def __init__(self, in_ch_left, in_ch_down, in_ch_right,top):
        super(BA1, self).__init__()
        self.CoordAttention = CoordAttention(256, 256,16)
        self.conv0 = nn.Conv2d(in_ch_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_ch_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_ch_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(top, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        #The above ops are used to reduce channels.left:low down:high right:global

        self.conv_d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_r = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(256 * 3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)



    def forward(self, left, down, right,top):
        
        top = F.relu(self.bn4(self.conv4(top)), inplace=True)  # 256 channels  
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels     
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d(down)
        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_1 * left, inplace=True)

        # z3
        right_1 = self.conv_r(right)
        if right_1.size()[2:] != left.size()[2:]:
            right_1 = F.interpolate(right_1, size=left.size()[2:], mode='bilinear')
        z3 = F.relu(right_1 * left, inplace=True)



        out = torch.cat((z1, z2, z3), dim=1)       

        out=F.relu(self.bn3(self.conv3(out)), inplace=True)

        top_1=self.CoordAttention(top)
       
        out=out*top_1+out      
 
        return out               
        

    def initialize(self):
        weight_init(self)


""" Body_Aggregation2 Module """
class BA2(nn.Module):
    def __init__(self, in_ch):
        super(BA2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)


class ConvBn(nn.Sequential):
    """
    Cascade of 2D convolution and batch norm.
    """

    def __init__(self, in_ch, out_ch, kernel_size, padding=0, dilation=1):
        super(ConvBn, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

    def initialize(self):
        weight_init(self)
        
class RFBblock(nn.Module):
    def __init__(self,in_ch,residual=False):
        super(RFBblock, self).__init__()
        inter_c = in_ch // 4
        self.branch_0 =  nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0)
        self.branch_11 = nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0)
        self.branch_12 = nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1)

        self.branch_21 = nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0)
        self.branch_22 = nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1)
        self.branch_23 = nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=2, padding=2)
                    
        self.branch_31 = nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0)
        self.branch_32 = nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=5, stride=1, padding=2)
        self.branch_33 = nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=3, padding=3)
                    
        self.residual= residual

    def forward(self,x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_11(x)
        x_1 = self.branch_12(x_1)
       
        x_2 = self.branch_21(x)
        x_2 = self.branch_22(x_2)
        x_2 = self.branch_23(x_2)


        x_3 = self.branch_31(x)  
        x_3 = self.branch_32(x_3)  
        x_3 = self.branch_33(x_3)  


        out = torch.cat((x_0,x_1,x_2,x_3),1)
        if self.residual:
            out +=x 
        return out
    def initialize(self):
        weight_init(self)

class SPPLayer(nn.Module):
    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_4 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4),dim=1)
        return out


class ASPPPooling(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBn(in_ch, out_ch, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = F.adaptive_avg_pool2d(x, 1)
        h = F.relu(self.conv(h))
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h

    def initialize(self):
        weight_init(self)


class ASPP(nn.Module):
    def __init__(self, in_ch, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        self.conv1 = ConvBn(in_ch, out_channels, 1)
        self.conv_aspp1 = ConvBn(in_ch, out_channels, 3, atrous_rates[0], atrous_rates[0])
        self.conv_aspp2 = ConvBn(in_ch, out_channels, 3, atrous_rates[1], atrous_rates[1])
        self.conv_aspp3 = ConvBn(in_ch, out_channels, 3, atrous_rates[2], atrous_rates[2])
        self.conv_pool = ASPPPooling(in_ch, out_channels)
        self.conv2 = ConvBn(5 * out_channels, in_ch, 1)

    def forward(self, x):
        res = []
        res.append(F.relu(self.conv1(x)))
        res.append(F.relu(self.conv_aspp1(x)))
        res.append(F.relu(self.conv_aspp2(x)))
        res.append(F.relu(self.conv_aspp3(x)))
        res.append(F.relu(self.conv_pool(x)))
        out = torch.cat([a for a in res], dim=1)
        out = F.relu(self.conv2(out))

        out = F.dropout(out, p=0.5, training=self.training)
        return out

    def initialize(self):
        weight_init(self)


class Coarse_Net(nn.Module):
    def __init__(self, in_ch_list):
        super(Coarse_Net, self).__init__()

        self.conv5_1 = nn.Conv2d(in_ch_list[3], 256, 1)
        self.aspp5 = ASPP(256, [1, 2, 4])
  
        self.conv5_2 = nn.Conv2d(256, 64, 1)
        self.conv5_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(64, 256, 1)

        self.conv4_1 = nn.Conv2d(in_ch_list[2], 256, 1)
   
        self.aspp4 = ASPP(256, [2, 4, 8])
        self.conv45_1 = nn.Conv2d(256, 64, 1)
        self.conv45_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv45_3 = nn.Conv2d(64, 256, 1)
        self.conv45_4 = nn.Conv2d(256, 256, 1)

        self.conv3_1 = nn.Conv2d(in_ch_list[1], 256, 1)
      
        self.rfb3=RFBblock(256)

        self.conv345_1 = nn.Conv2d(256, 64, 1)
        self.conv345_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv345_3 = nn.Conv2d(64, 256, 1)
        self.conv345_4 = nn.Conv2d(256, 256, 1)

        self.conv345_123_1 = nn.Conv2d(256, 64, 1)
        self.conv345_123_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv345_123_3 = nn.Conv2d(64, 256, 1)
        self.conv345_123_4 = nn.Conv2d(256, 256, 1)

        self.conv2_1 = nn.Conv2d(in_ch_list[0], 64, 1)
        self.rfb2=RFBblock(64)
     
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 256, 1)


    def forward(self, x, out2, out3, out4, out5):
        out5_1 = F.relu(self.conv5_1(out5), inplace=False)
        out5_1 = self.aspp5(out5_1)
     
        out5_2 = F.relu(self.conv5_2(out5_1), inplace=False)
        out5_3 = F.relu(self.conv5_3(out5_2), inplace=False)
        out5_4 = self.conv5_4(out5_3)

        out5_4_i = F.interpolate(out5_4, size=out4.size()[2:], mode='bilinear', align_corners=False)
        out4_1 = self.conv4_1(out4)      
        out4_1=self.aspp4(out4_1)

        out45 = F.relu((out5_4_i + out4_1), inplace=False)
        out45_1 = F.relu(self.conv45_1(out45), inplace=False)
        out45_2 = F.relu(self.conv45_2(out45_1), inplace=False)
        out45_3 = self.conv45_3(out45_2)
        out45_4 = self.conv45_4(out45)

        out45_3_i = F.interpolate(out45_3, size=out3.size()[2:], mode='bilinear', align_corners=False)
        out45_4_i = F.interpolate(out45_4, size=out3.size()[2:], mode='bilinear', align_corners=False)
        out3_1 = self.conv3_1(out3)
        out3_1=self.rfb3(out3_1)
   
        out345 = F.relu((out45_3_i + out45_4_i + out3_1), inplace=False)
        out345_1 = F.relu(self.conv345_1(out345), inplace=False)
        out345_2 = F.relu(self.conv345_2(out345_1), inplace=False)
        out345_3 = self.conv345_3(out345_2)
        out345_4 = self.conv345_4(out345)

        out345_123 = F.relu((out345_3 + out345_4), inplace=False)
        out345_123_1 = F.relu(self.conv345_123_1(out345_123), inplace=False)
        out345_123_2 = F.relu(self.conv345_123_2(out345_123_1), inplace=False)
        out345_123_3 = self.conv345_123_3(out345_123_2)
        out345_123_4 = self.conv345_123_4(out345_123)

        out345_123_3_i = F.interpolate(out345_123_3, size=out2.size()[2:], mode='bilinear', align_corners=False)
        out345_123_4_i = F.interpolate(out345_123_4, size=out2.size()[2:], mode='bilinear', align_corners=False)
        out345_4_i = F.interpolate(out345_4, size=out2.size()[2:], mode='bilinear', align_corners=False)

        out2_1 = F.relu(self.conv2_1(out2), inplace=False)
        out2_1=self.rfb2(out2_1)
     
        out2_2 = F.relu(self.conv2_2(out2_1), inplace=False)
        out2_3 = self.conv2_3(out2_2)

        out2 = F.relu((out345_123_3_i + out345_123_4_i + out345_4_i + out2_3), inplace=False)

        return out2,out345,out45
    def initialize(self):
        weight_init(self)


class SENet(nn.Module):
    def __init__(self, cfg):
        super(SENet, self).__init__()
        self.aspp = ASPP(256, [1, 2, 4])
        self.cfg = cfg
        self.bkbone = ResNet()
        #self.sync_bn = torch.nn.SyncBatchNorm(num_features, eps=1e-05, momentum=0.1, affine=True, 
        #                          track_running_stats=True)


        self.ca45 = CALayer(2048, 2048)
        self.ca35 = CALayer(2048, 2048)
        self.ca25 = CALayer(2048, 2048)
        self.ca55 = CALayer(256, 2048)
        self.ca_c2 = CALayer(256, 256)
        self.ca_c3 = CALayer(256, 256)
        self.ca_c4 = CALayer(256, 256)


        self.ba1_45 = BA1(1024, 256, 256,256)
        self.ba1_34 = BA1(512, 256, 256,256)
        self.ba1_23 = BA1(256, 256, 256,256)

  
        self.ba2_4 = BA2(256)
        self.ba2_3 = BA2(256)
        self.ba2_2 = BA2(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)


        self.Coarse_Net = Coarse_Net([256, 512, 1024, 2048])
      
        self.conv_o = nn.Conv2d(2048, 256, 1)
        self.conv_p = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_q = nn.Conv2d(256, 2048, 1)
        self.bn = nn.BatchNorm2d(1)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn2048 = nn.BatchNorm2d(2048)

        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)

        # edge
        out2_coarse_a, out345,out45= self.Coarse_Net(x, out2, out3, out4, out5_)

        out5_f = F.relu(self.conv_o(out5_))  # 256 1*1
        out5_fp = self.aspp(out5_f)  # 256
        out5_f = self.bn2048(self.conv_q(out5_fp))  # 1*1

        out5_f = F.sigmoid(out5_f)
        out5_ = out5_ + out5_f * out5_

        # CA
        out4_a = self.ca45(out5_, out5_)
        out3_a = self.ca35(out5_, out5_)
        out2_a = self.ca25(out5_, out5_)

        
        out345= self.ca_c3(out345, out345)
        out45= self.ca_c4(out45, out45)
        out2_coarse_a =self.ca_c2(out2_coarse_a, out2_coarse_a)


        out5 = out5_fp
        out4 = self.ba2_4(self.ba1_45(out4, out5, out4_a,out45))
        out3 = self.ba2_3(self.ba1_34(out3, out4, out3_a,out345))
        out2 = self.ba2_2(self.ba1_23(out2, out3, out2_a,out2_coarse_a))       
      
        
        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)
        out2 = self.linear2(out2)        
        out2_coarse_a = self.linear1(out2_coarse_a)
       

        out5 = F.interpolate(out5, size=x.size()[2:], mode='bilinear')
        out4 = F.interpolate(out4, size=x.size()[2:], mode='bilinear')
        out3 = F.interpolate(out3, size=x.size()[2:], mode='bilinear')
        out2 = F.interpolate(out2, size=x.size()[2:], mode='bilinear')
        out2_coarse_a = F.interpolate(out2_coarse_a, size=x.size()[2:], mode='bilinear')


        return out2, out3, out4, out5,out2_coarse_a#,out345,out45#, out2_a  # , out3_coarse, out4_coarse, out5_coarse

    def initialize(self):
        if self.cfg.snapshot:  # 监控snapshot状态
            #try:
            self.load_state_dict(torch.load(self.cfg.snapshot,map_location='cuda:0'))
            # except:
            #     print("Warning: please check the snapshot file:", self.cfg.snapshot)
            #     pass
        else:
            weight_init(self)


