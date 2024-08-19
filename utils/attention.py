# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 19:30
# @Author  : wwj
# @FileName: attention.py
# @Software: PyCharm
# @Brief:
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------通道注意力---------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l1', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            # sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class lca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=5):
        super(lca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False, dilation=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        b, n, c = x.size()
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.conv(y)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]

        return x


# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, channel, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         print(x.shape)
#         b, c, h, w = x.size()
#         print(x.shape)
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#         print(y.shape)
#
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)

class SKLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, reduction=4):
        super(SKLayer, self).__init__()
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, 1, groups=groups, bias=False)
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, 2, groups=groups, bias=False, dilation=2)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.bn5 = nn.BatchNorm2d(out_planes)
        self.active = nn.ReLU6(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc1 = nn.Conv2d(in_planes, out_planes // reduction, 1, bias=False)
        self.bn_fc1 = nn.BatchNorm2d(out_planes // reduction)
        self.conv_fc2 = nn.Conv2d(out_planes // reduction, 2 * out_planes, 1, bias=False)
        self.D = out_planes

    def forward(self, x):
        d1 = self.conv3(x)
        d1 = self.bn3(d1)
        d1 = self.active(d1)

        d2 = self.conv5(x)
        d2 = self.bn5(d2)
        d2 = self.active(d2)

        d = self.avg_pool(d1) + self.avg_pool(d2)
        d = F.relu(self.bn_fc1(self.conv_fc1(d)))
        d = self.conv_fc2(d)
        d = torch.unsqueeze(d, 1).view(-1, 2, self.D, 1, 1)
        d = F.softmax(d, 1)
        d1 = d1 * d[:, 0, :, :, :].squeeze(1)
        d2 = d2 * d[:, 1, :, :, :].squeeze(1)
        d = d1 + d2
        return d


# -----------------空间注意力---------------------
class CBAM_Module(nn.Module):

    def __init__(self, channels, kernel_size, reduction=4):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, int(channels // reduction), kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(int(channels // reduction), channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        if kernel_size == 3:
            padding = 1
        if kernel_size == 5:
            padding = 2
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttentionModule, self).__init__()
        # self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        # self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


## ZAM ##
class ZeroChannelAttention(nn.Module):
    def __init__(self):
        super(ZeroChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Hardswish()
        # self.sigmoid = nn.ReLU()
        self.sigmoid = nn.Hardsigmoid()
        # swish()
        # Relu()

    def forward(self, x):
        return self.sigmoid(self.avg_pool(x) + self.max_pool(x))


class ZeroSpatialAttention(nn.Module):
    def __init__(self):
        super(ZeroSpatialAttention, self).__init__()

        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Hardswish()
        # self.sigmoid = nn.ReLU()
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(avg_out + max_out)


class ZAM(nn.Module):
    def __init__(self, use_skip_connection=False):
        super(ZAM, self).__init__()

        self.ca = ZeroChannelAttention()
        self.sa = ZeroSpatialAttention()
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        out = x + x * self.ca(x) if self.use_skip_connection else x * self.ca(x)
        out = out + out * self.sa(out) if self.use_skip_connection else out * self.sa(out)

        return out
