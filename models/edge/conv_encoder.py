import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .layers import LayerNorm
import ops


class ConvEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7,args = None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn0 = nn.BatchNorm2d(dim)

        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.args = args



    def forward(self, x):
        # glo = torch.cat((self.gcfm(x),x),dim=1)
        y = self.gcfm(x)
        input = x   # 只能是一个参数，为什么这里改成输入 y 和  x
        x = self.dwconv(x)

        x= ops.gating_op(x, self.args)
        # x = torch.nn.functional.interpolate(x , size=None, scale_factor=1, mode='bilinear', align_corners=None, recompute_scale_factor=None)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def gcfm(self,x):
        f = F.relu(self.bn0(self.conv0(x)), inplace=True)  # 也都是 1*1 卷积
        y = x.mean(dim=(2, 3), keepdim=True)  # 相当于平均池化
        y = F.relu(self.conv0(y), inplace=True)  # 1*1
        y = torch.sigmoid(self.conv0(y))  # 1*1
        return f * y


class ConvEncoderBNHS(nn.Module):
    """
        Conv. Encoder with Batch Norm and Hard-Swish Activation
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.Hardswish()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
