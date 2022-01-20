import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from model.vgg import VGG


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class CorrelationModule(nn.Module):
    def  __init__(self, all_channel=128, all_dim=32):
        super(CorrelationModule, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel,bias = False)
        self.channel = all_channel
        self.dim = all_dim*all_dim
        self.gate_1 = nn.Conv2d(all_channel, 1, kernel_size=1, bias = False)
        self.gate_2 = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = DSConv3x3(all_channel, all_channel, stride=1)
        self.conv2 = DSConv3x3(all_channel, all_channel, stride=1)
        self.conv_fusion = DSConv3x3(all_channel*2, all_channel, stride=1)
        self.pred = nn.Conv2d(all_channel, 1, kernel_size=1, bias = True)
        # self.pred2 = nn.Conv2d(all_channel, 1, kernel_size=1, bias = True)

    def forward(self, exemplar, query): # exemplar: low resolution, query: high resolution
        fea_size = query.size()[2:]
        exemplar = F.interpolate(exemplar, size=fea_size, mode="bilinear", align_corners=True)
#		 #all_dim = exemplar.shape[1]*exemplar.shape[2]
        all_dim = fea_size[0]*fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim) #N,C,H*W
        query_flat = query.view(-1, self.channel, all_dim)
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  #batchsize x dim x num
        exemplar_corr = self.linear_e(exemplar_t) #
        A = torch.bmm(exemplar_corr, query_flat)

        A1 = F.softmax(A.clone(), dim = 1) #
        B = F.softmax(torch.transpose(A,1,2),dim=1)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        exemplar_mask = self.gate_1(exemplar_att)
        exemplar_mask = self.gate_s(exemplar_mask)
        exemplar_att = exemplar_att * exemplar_mask
        exemplar_out = self.conv1(exemplar_att + exemplar)
        # pred1: low resolution
        # pred1 = self.pred1(exemplar_out)

        query_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])
        query_mask = self.gate_2(query_att)
        query_mask = self.gate_s(query_mask)
        query_att = query_att * query_mask
        query_out = self.conv1(query_att + query)
        # pred2: high resolution
        # pred2 = self.pred2(query_out)

        pred = self.pred(self.conv_fusion(torch.cat([exemplar_out,query_out],1)))
        return pred

# DLDblock is DLRB in our published paper
class DLDblock(nn.Module):
    def __init__(self,channel):
        super(DLDblock, self).__init__()

        # self.stage11 = DSConv3x3(channel, channel, stride=1)
        self.stage12 = DSConv3x3(channel, channel, stride=1, dilation=2)
        self.fuse1 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)

        # self.stage21 = DSConv3x3(channel, channel, stride=1)
        self.stage22 = DSConv3x3(channel, channel, stride=1, dilation=4)
        self.fuse2 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)

        # self.stage31 = DSConv3x3(channel, channel, stride=1)
        self.stage32 = DSConv3x3(channel, channel, stride=1, dilation=6)
        self.fuse3 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)

    def forward(self, x):
        # x11 = self.stage11(x)
        x12 = self.stage12(x)
        x1 = self.fuse1(x+x12)

        # x21 = self.stage21(x1)
        x22 = self.stage22(x1)
        x2 = self.fuse2(x+x1+x22)

        # x31 = self.stage31(x2)
        x32 = self.stage32(x2)
        x3 = self.fuse3(x+x1+x2+x32)

        return x3

class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                # nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)

class prediction_decoder(nn.Module):
    def __init__(self):
        super(prediction_decoder, self).__init__()

        #self.adpative_weight = nn.Parameter(torch.ones(channel, dtype=torch.float32), requires_grad=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.ca_x1 = ChannelAttention(64)
        self.ca_x2 = ChannelAttention(128)
        self.sa_x1 = SpatialAttention()
        self.sa_x2 = SpatialAttention()

	# DLDblock is DLRB in our published paper
        self.decoder3 = nn.Sequential(
                DLDblock(256),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DSConv3x3(256, 128, stride=1)
                )
        self.s3 = SalHead(128)

        self.decoder2 = nn.Sequential(
                DSConv3x3(256, 128, stride=1),
                DLDblock(128),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DSConv3x3(128, 64, stride=1)
                )
        self.s2 = SalHead(64)

        self.decoder1 = nn.Sequential(
                DSConv3x3(128, 64, stride=1),
                DLDblock(64),
                )
        self.s1 = SalHead(64)

    def forward(self, pred, x1, x2, x3):
        x1_ca = x1.mul(self.ca_x1(x1))
        x1_sa = x1_ca.mul(self.sa_x1(x1_ca))
        x1 = x1 + x1_sa

        x2_ca = x2.mul(self.ca_x2(x2))
        x2_sa = x2_ca.mul(self.sa_x2(x2_ca))
        x2 = x2 + x2_sa

        x3 = x3+torch.mul(x3, self.upsample2(pred))
        x3_decoder = self.decoder3(x3)
        s3 = self.s3(x3_decoder)

        x2_decoder = self.decoder2(torch.cat([x3_decoder,x2],1))
        s2 = self.s2(x2_decoder)

        x1_decoder = self.decoder1(torch.cat([x2_decoder,x1],1))
        s1 = self.s1(x1_decoder)

        return s1, s2, s3


class CorrelationModel_VGG(nn.Module):
    def __init__(self, channel=128):
        super(CorrelationModel_VGG, self).__init__()
        # Backbone model
        self.vgg = VGG('rgb')

        # Normalized the channel number to 128
        self.conv5 = DSConv3x3(512, channel, stride=1)
        self.conv4 = DSConv3x3(512, channel, stride=1)

        # CorrelationModel
        self.Corr45 = CorrelationModule(channel, 32)

        self.prediction_decoder = prediction_decoder()

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x_rgb):
        x1 = self.vgg.conv1(x_rgb)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x4 = self.vgg.conv4(x3)
        x5 = self.vgg.conv5(x4)

        x4_rgb = self.conv4(x4)
        x5_rgb = self.conv5(x5)

        pred54 = self.Corr45(x5_rgb,x4_rgb)
        pred54_up = self.upsample8(pred54)
        pre_pred = self.sigmoid(pred54)

        s1, s2, s3 = self.prediction_decoder(pre_pred, x1, x2, x3)

        s3_up = self.upsample2(s3)

        return s1, s2, s3_up, pred54_up, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3_up), self.sigmoid(pred54_up)
