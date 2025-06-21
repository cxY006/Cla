import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn

# from tensorpack.tfutils.tower import get_current_tower_context
# from tensorpack.models import Conv2D


def gating_op(input_, option):
    return Key_Region_Adaptive_Enhancement(input_, option)
    # if option.method_name == 'CAM':
    #     output = input_
    # elif option.method_name == 'ADL':
    #     output = attention_based_dropout(input_, option)
    # else:
    #     raise KeyError("Unavailable method: {}".format(option.method_name))
    #
    # return output


def Key_Region_Adaptive_Enhancement(input_, option):

    key_enhance = KeyObjectEnhancement(topk=5)

    def _get_importance_map(attention):
        return tf.sigmoid(attention)

    def _get_drop_mask(attention, drop_thr):
        max_val = tf.reduce_max(attention, axis=[1, 2, 3], keepdims=True)
        thr_val = max_val * drop_thr
        return tf.cast(attention < thr_val, dtype=tf.float32, name='drop_mask')

    def _select_component(importance_map, drop_mask, drop_prob):
        random_tensor = tf.random.uniform([], drop_prob, 1. + drop_prob)
        binary_tensor = tf.cast(tf.floor(random_tensor), dtype=tf.float32)
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    # ctx = get_current_tower_context()
    # is_training = ctx.is_training
    is_training = True


    # drop_prob = 1 - option.adl_keep_prob
    # drop_thr = option.adl_threshold
    drop_prob = 1 - 0.25
    drop_thr = 0.5

    if is_training:
        input_1 = tf.convert_to_tensor(input_.cpu().detach().numpy())
        attention_map = tf.reduce_mean(input_1, axis=1, keepdims=True)
        key_attention = key_enhance(attention_map)
        # importance_map = _get_importance_map(attention_map)
        drop_mask = _get_drop_mask(attention_map, drop_thr)
        selected_map = _select_component(key_attention, drop_mask, drop_prob)
        # print("selected_map shape before reshape:", selected_map.shape)
        # selected_map = selected_map[:, :80, :, :]
        min_dim1 = min(input_1.shape[1], selected_map.shape[1])
        output = input_1[:, :min_dim1] * selected_map[:, :min_dim1]
        return torch.Tensor(output.numpy()).to('cuda')

    else:
        return input_


class KeyObjectEnhancement(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        self.topk = topk  # 保留最重要的topk个区域

    def forward(self, x):
        # 计算注意力热力图
        att_map = tf.reduce_mean(tf.abs(x), axis=1, keepdims=True)  # [B,1,H,W]

        # 找出最重要的topk个区域
        B, _, H, W = att_map.shape
        # _, indices = att_map.view(B, -1).topk(self.topk, dim=1)
        reshaped_att = tf.reshape(att_map, [B, -1])  # 替代 att_map.view(B, -1)
        _, indices = tf.math.top_k(reshaped_att, k=self.topk)
        # mask = torch.zeros_like(att_map.view(B, -1))
        # mask.scatter_(1, indices, 1.0)
        # mask = mask.view(B, 1, H, W)
        # mask = tf.zeros_like(tf.reshape(att_map, [B, -1]))  # 替代 view(B, -1)

        mask = tf.one_hot(indices, depth=H * W, dtype=tf.float32)  # 形状 [B, topk, H*W]
        mask = tf.reduce_max(mask, axis=1)  # 形状 [B, H*W]
        mask = tf.reshape(mask, [B, H, W])  # 或 [B, 1, H, W]

        # 增强关键区域，弱化其他区域
        return x * (mask + 0.3)  # 关键区域×1.3，其他×0.3

# def convnormrelu(x, name, chan, kernel_size=3, padding='SAME'):
#     x = Conv2D(name, x, chan, kernel_size=kernel_size, padding=padding)
#     x = tf.nn.relu(x, name=name + '_relu')
#     return x
