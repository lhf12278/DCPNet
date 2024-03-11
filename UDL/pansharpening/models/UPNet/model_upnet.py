import numpy
import torch.nn as nn
from torch import  einsum
import torchvision.models as models
import torch
import torch.nn.functional as F
import os
import cv2
import math
from torchvision.ops import DeformConv2d
from UDL.pansharpening.models.UPNet.pytorch_ssim import ssim, _ssim
import numpy as np
import UDL.pansharpening.models.UPNet.SwinT as swin
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchvision import transforms
from einops import rearrange

class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_output = self.conv(x)
        gate_output = self.gate(x)
        gated_output = self.sigmoid(gate_output) * conv_output
        return gated_output

class DWConv(nn.Module):
    def __init__(self, channels):
        super(DWConv, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels*3, kernel_size=1, bias=True)
        self.deptconv = nn.Conv2d(channels*3, channels*3, kernel_size=3, stride=1, padding=1, groups=channels*3, bias=True)
        # self.pointconv = nn.Conv2d(channels*3, channels*3, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.deptconv(out)
        # out = self.pointconv(out)
        return out

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        # qk_scale=1/math.sqrt(dim),
        attn_drop=0.,
        proj_drop=0.,
        qkv_bias=True
    ):
        super().__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dw_conv = DWConv(dim)
        self.attn_sd = nn.Softmax(dim=-1)
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, pan, ms):
        b, c, h, w = pan.shape
        qkv1 = self.dw_conv(pan)
        q1, k1, v1 = qkv1.chunk(3, dim=1)

        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        qkv2 = self.dw_conv(ms)
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q2 @ k1.transpose(2, 3)) * self.temperature
        attn = self.attn_sd(attn)
        out = (attn @ v1)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj_conv(out)
        out = out + pan
        return out

class Attention_hw(nn.Module):
    def __init__(self, dim, num_heads, window_size, attn_drop=0., proj_drop=0., qkv_bias=True):
        super(Attention_hw, self).__init__()

        self.num_heads = num_heads
        # self.scale = 1 / math.sqrt(dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dw_conv = DWConv(dim)
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.window_size = window_size
        self.attn_sd = nn.Softmax(dim = -1)

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.dw_conv(x)
        q,k,v = qkv.chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x ws1) (y ws2) -> (b x y) h (ws1 ws2) d', h=self.num_heads, ws1=self.window_size,
                                          ws2=self.window_size), (q, k, v))
        q = q * self.temperature
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = self.attn_sd(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, '(b x y) head (ws1 ws2) d -> b (head d) (x ws1) (y ws2)',
                        x=h // self.window_size, y=w // self.window_size, head=self.num_heads, ws1=self.window_size, ws2=self.window_size)
        out = self.proj_conv(out)
        out = out + x
        return out

class Attention_cc(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super(Attention_cc, self).__init__()

        self.num_heads = num_heads
        # self.temperature = 1 / math.sqrt(dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dw_conv = DWConv(dim)
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.dw_conv(x)
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(2, 3)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj_conv(out)
        out = out + x
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
        nn.Conv2d(channel, channel // reduction, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel // reduction, channel, 1, bias=False),
        nn.Sigmoid()
        )

    def forward(self, ms):
        b, c, _, _ = ms.size()
        y = self.avg_pool(ms)
        y = self.fc(y)
        out = ms * y
        return out

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.L_relu = nn.LeakyReLU(negative_slope=0.25)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, f):
        out = self.conv1(f)
        out = self.L_relu(out)
        out = self.conv2(out)
        out = out + f
        return out

class Channel_Rearrange(nn.Module):
    def __init__(self):
        super(Channel_Rearrange, self).__init__()

    def forward(self, ms, pan):
        list = []
        B1, C1, H1, W1 = ms.shape
        B2, C2, H2, W2 = pan.shape

        for i in range(C1):
            ms1 = ms[:, i, :, :].unsqueeze(1)
            pan1 = pan[:, i, :, :].unsqueeze(1)
            f1 = torch.cat((ms1, pan1), 1)
            list.append(f1)
        out = torch.cat(list, dim=1)
        return out

class M_FEB(nn.Module):
    def __init__(self):
        super(M_FEB,self).__init__()

        self.st1 = swin.SwinT(32)
        self.st2 = swin.SwinT(32)
        self.sts = nn.Sequential(
            self.st1,
            self.st2,
        )
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.L_relu = nn.LeakyReLU(negative_slope=0.25)
        self.cross = CrossAttention(32, 4)

    def forward(self, ms, pan):
        ms1 = self.sts(ms)
        ms2 = self.L_relu(self.conv1(ms1))
        mp = self.cross(pan, ms)
        out = ms2 + mp
        return out

class P_FEB(nn.Module):
    def __init__(self, channels=32):
        super(P_FEB,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv4 = nn.Conv2d(in_channels=channels * 3, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.L_relu = nn.LeakyReLU(negative_slope=0.25)
        self.P_PCA1 = CrossAttention(32, 4)
        self.P_PCA2 = CrossAttention(32, 4)

    def forward(self, pan):
        out1 = self.L_relu(self.conv1(pan))
        out2 = self.L_relu(self.conv2(pan))
        out3 = self.L_relu(self.conv3(pan))
        CA1_2 = self.P_PCA1(out2, out1)
        CA1_3 = self.P_PCA2(out3, out1)
        out4 = torch.cat([out1, CA1_2, CA1_3], 1)
        out5 = self.conv4(out4)
        out = out5 + pan
        return out

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion,self).__init__()

        self.attn_hw = Attention_hw(64, 4, 8)
        self.attn_cc = Attention_cc(64, 4)
        self.conv = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)
        self.arrange = Channel_Rearrange()

    def forward(self, x):
        out1 = self.attn_hw(x)
        out2 = self.attn_cc(x)
        out3 = self.arrange(out1, out2)
        out4 = self.conv(out3)
        return out4

class Pre_net(nn.Module):
    def __init__(self, spectral_num):
        super(Pre_net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=spectral_num, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)

        self.cross = CrossAttention(32, 4)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.L_relu = nn.LeakyReLU(negative_slope=0.25)

        self.CA = SELayer(channel=32, reduction=8)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, pan, ms):
        pan1 = self.conv1(pan)
        ms1 = self.conv2(ms)
        ms2 = F.pixel_shuffle(ms1, 4)
        ms3 = self.CA(ms2)
        ms4 = self.L_relu(self.conv4(ms3))
        pan2 = self.cross(pan1, ms2)
        pan3 = self.L_relu(self.conv3(pan2))
        return pan3, ms4

class PAN_net(nn.Module):
    def __init__(self):
        super(PAN_net, self).__init__()

        self.pan_feb1 = P_FEB()
        self.pan_feb2 = P_FEB()
        self.pan_feb3 = P_FEB()
        self.pan_feb4 = P_FEB()

    def forward(self, pan):
        panf1 = self.pan_feb1(pan)
        panf2 = self.pan_feb2(panf1)
        panf3 = self.pan_feb3(panf2)
        panf4 = self.pan_feb4(panf3)
        return panf1, panf2, panf3, panf4

class MS_net(nn.Module):
    def __init__(self, spectral_num):
        super(MS_net, self).__init__()

        self.ms_feb1 = M_FEB()
        self.ms_feb2 = M_FEB()
        self.ms_feb3 = M_FEB()
        self.ms_feb4 = M_FEB()

        self.rb1 = ResBlock(32)
        self.rb2 = ResBlock(32)
        self.rb3 = ResBlock(32)
        self.rb4 = ResBlock(32)
        self.rb5 = ResBlock(32)
        self.ms_rbs = nn.Sequential(
            self.rb1,
            self.rb2,
            self.rb3,
            self.rb4,
            self.rb5,
        )
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, ms, pan1, pan2, pan3, pan4):
        msf1 = self.ms_feb1(ms, pan1)
        msf2 = self.ms_feb2(msf1, pan2)
        msf3 = self.ms_feb3(msf2, pan3)
        msf4 = self.ms_feb4(msf3, pan4)
        msf = self.ms_rbs(msf4)
        msup = self.conv5(msf)
        return msup, msf1, msf2, msf3, msf4

class Fusion_net(nn.Module):
    def __init__(self, spectral_num):
        super(Fusion_net, self).__init__()

        self.arrange = Channel_Rearrange()
        self.fub1 = Fusion()
        self.fub2 = Fusion()
        self.fub3 = Fusion()
        self.fub4 = Fusion()

        self.gate = GatedConv(64,64)

        self.frb1 = ResBlock(64)
        self.frb2 = ResBlock(64)
        self.frb3 = ResBlock(64)
        self.frb4 = ResBlock(64)
        self.frb5 = ResBlock(64)
        self.f_rbs = nn.Sequential(
            self.frb1,
            self.frb2,
            self.frb3,
            self.frb4,
            self.frb5,
        )
        self.reconv = nn.Conv2d(in_channels=64, out_channels=spectral_num, kernel_size=1, stride=1, padding=0, bias=True)
        self.convf1 = nn.Conv2d(in_channels=64*4, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, panf1, panf2, panf3, panf4, msf1, msf2, msf3, msf4):
        f1 = self.arrange(msf1, panf1)
        f2 = self.arrange(msf2, panf2)
        f3 = self.arrange(msf3, panf3)
        f4 = self.arrange(msf4, panf4)
        fu1 = self.fub1(f1)
        fu1 = self.gate(fu1)
        fu2 = self.fub2(f2)
        fu2 = self.gate(fu2)
        fu3 = self.fub3(f3)
        fu3 = self.gate(fu3)
        fu4 = self.fub4(f4)
        fu4 = self.gate(fu4)
        fi = torch.cat([fu1,fu2,fu3,fu4], dim=1)
        fi = self.convf1(fi)
        fi = self.f_rbs(fi)
        sr = self.reconv(fi)
        return sr

class UPNet(nn.Module):
    def __init__(self, spectral_num, criterion):
        super(UPNet, self).__init__()

        self.spectral_num = spectral_num
        self.criterion = criterion
        self.pre_net = Pre_net(spectral_num)
        self.pan_net = PAN_net()
        self.ms_net = MS_net(spectral_num)
        self.fusion_net = Fusion_net(spectral_num)

    def forward(self, pan, ms):
        pan1, ms1 = self.pre_net(pan, ms)

        panf1, panf2, panf3, panf4 = self.pan_net(pan1)

        msup, msf1, msf2, msf3, msf4 = self.ms_net(ms1, panf1, panf2, panf3, panf4)

        sr = self.fusion_net(panf1, panf2, panf3, panf4, msf1, msf2, msf3, msf4)
        return sr, msup

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()

        sr, msup = self(pan, ms)

        loss1 = self.criterion(sr, gt, *args, **kwargs)['loss']
        loss2 = self.criterion(msup, gt, *args, **kwargs)['loss']

        loss =  loss1 + 0.5*loss2
        log_vars.update(loss=loss.item())

        return {'loss': loss, 'log_vars': log_vars}

    def val_step(self, data, *args, **kwargs):
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()
        sr,msup = self(pan, ms)

        return sr, gt