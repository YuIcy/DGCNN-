#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_semseg_s3dis(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_s3dis, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x
        

class DGCNN_Adaptor(nn.Module):
    def __init__(self, opt):
        super(DGCNN_Adaptor, self).__init__()
        self.opt = opt
        self.k = opt.k
        channels = 64
        
        # 1. Head / Embedding Layers
        # 输入维度是 opt.in_channels (例如 9)
        self.conv1 = nn.Sequential(nn.Conv2d(18, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.LeakyReLU(negative_slope=0.2))

        # 2. Fusion Layer
        # 拼接 net_1, net_2, net_3，所以是 channels * 3
        self.fusion_block = nn.Sequential(nn.Conv2d(channels * 3, 1024, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(1024),
                                          nn.LeakyReLU(negative_slope=0.2))

        # 3. Prediction Header
        # 拼接全局特征 (1024) 和 局部特征 (channels * 3)
        fusion_dims = 1024 + channels * 3
        self.prediction = nn.Sequential(
            nn.Conv1d(fusion_dims, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=opt.dropout),
            nn.Conv1d(256, 13, kernel_size=1, bias=True)
        )

    def forward(self, inputs):
        # 统一输入格式为 (B, C, N)，如果输入是 (B, C, N, 1) 则去掉最后一维
        if inputs.dim() == 4:
            inputs = inputs.squeeze(-1)
        
        batch_size = inputs.size(0)
        num_points = inputs.size(2)

        # 第一层：使用坐标/颜色等原始特征做 KNN
        # 对应原代码 point_cloud[:, :, 6:]，如果 in_channels 为 9，通常 0:3 是坐标
        # 这里为了通用，可以根据需求指定 KNN 的特征范围，这里假设用前3维坐标
        idx = knn(inputs[:, 0:3, :], k=self.k) 

        # Block 1
        x = get_graph_feature(inputs, k=self.k, idx=idx)
        x = self.conv1(x)
        x = self.conv2(x)
        net_1 = x.max(dim=-1, keepdim=False)[0]

        # Block 2
        x = get_graph_feature(net_1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        net_2 = x.max(dim=-1, keepdim=False)[0]

        # Block 3
        x = get_graph_feature(net_2, k=self.k)
        x = self.conv5(x)
        net_3 = x.max(dim=-1, keepdim=False)[0]

        # 特征拼接
        feats = torch.cat((net_1, net_2, net_3), dim=1) # (B, C*3, N)

        # 全局特征聚合
        # 对应 DenseDeepGCN 的 max_pool2d 逻辑
        fusion = self.fusion_block(feats.unsqueeze(-1)) # (B, 1024, N, 1)
        fusion = F.adaptive_max_pool2d(fusion, (1, 1)) # (B, 1024, 1, 1)
        
        # 广播全局特征
        fusion = fusion.repeat(1, 1, num_points, 1).squeeze(-1) # (B, 1024, N)
        
        # 拼接全局和局部特征进行预测
        out = torch.cat((fusion, feats), dim=1) # (B, 1024 + C*3, N)
        
        # 输出形状为 (B, n_classes, N)
        return self.prediction(out)
    

def gumbel_sigmoid(logits, tau=1.0, hard=False,training = True):
    """
    可微的二值采样：
    - Forward: 输出近似 0 或 1 的值 (如果 hard=True，则严格输出 0/1)
    - Backward: 梯度可传导回 logits
    """
    if not training:
        return (logits >= 0).float()
        k_retain = int(logits.size(-1) * 0.5)  # 保留一半邻居
        # 找第 k_retain 大的值作为动态阈值
        topk_val, _ = torch.topk(logits, k=k_retain, dim=-1)
        threshold = topk_val[:, :, :, -1].unsqueeze(-1)
        
        # 生成硬 Mask
        y_hard = (logits >= threshold).float()
        return y_hard
    # 生成 Gumbel 噪声
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = torch.sigmoid(gumbels)

    if hard:
        # Straight-Through Estimator (STE)
        # 前向传播是硬截断(>0.5为1)，反向传播用软梯度的 y_soft
        index = (y_soft >= 0.5).float()
        y_hard = index - y_soft.detach() + y_soft
        return y_hard
    else:
        return y_soft

class GSL_Block(nn.Module):
    def __init__(self,opt, in_channels, out_channels, k, transform_layers, mode='full'):
        """
        mode: 
          - 'gating_only': 只有硬门控剪枝 (Gumbel-Sigmoid + Max Pooling) -> 类似 DGCNN 但能删邻居
          - 'attn_only':   只有软注意力 (Softmax + Weighted Sum) -> 类似 Point Transformer
          - 'full':        既有剪枝又有注意力 (Gumbel + Masked Softmax + Sum) -> 你的完整方法
        """
        super(GSL_Block, self).__init__()
        self.k = k
        self.mode = mode
        self.trans = transform_layers
        self.out_channels = out_channels

        # 1. 定义网络组件 (根据模式按需定义，节省显存)
        
        # [Gating 组件]: 用于 'gating_only' 和 'full'
        if self.mode in ['gating_only', 'full']:
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                nn.Conv2d(8, 1, kernel_size=1, bias=True)
            )
            # nn.init.constant_(self.gate_net[-1].bias, 5.0) 

        # [Attention 组件]: 用于 'attn_only' 和 'full'
        if self.mode in ['attn_only', 'full']:
            self.attn_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                nn.Conv2d(8, 1, kernel_size=1, bias=True)
            )

    def forward(self, x, rel_pos=None):
        # x: (B, 2*Cin, N, K) 边缘特征
        
        # 先进行通用的特征变换 (MLP)
        trans_feat = self.trans(x) # (B, Cout, N, K)

        # ==========================================================
        # 模式 A: 仅门控 (Gating Only) - "硬剪枝 + Max Pooling"
        # ==========================================================
        if self.mode == 'gating_only':
            # 1. 计算 Mask (0 或 1)
            gate_logits = self.gate_net(x)
            mask = gumbel_sigmoid(gate_logits, tau=1.0, hard=True,training=self.training) # (B, 1, N, K)
            
            # 2. 聚合逻辑
            # 我们希望被 Mask 掉的邻居完全不参与 Max Pooling
            # 方法: 将 Mask=0 的位置填充为负无穷 (-1e9)
            # 注意: mask 广播到 trans_feat 的维度
            masked_feat = trans_feat.masked_fill(mask < 0.5, -1e9)
            
            # 3. Max Pooling (DGCNN 的标准聚合方式)
            out = masked_feat.max(dim=-1)[0]
            return out

        # ==========================================================
        # 模式 B: 仅注意力 (Attention Only) - "软权重 + Sum"
        # ==========================================================
        elif self.mode == 'attn_only':
            # 1. 计算原始 Attention 分数
            attn_logits = self.attn_net(x)
            
            # 2. 标准 Softmax 全局归一化 (所有邻居都参与)
            attn_weights = F.softmax(attn_logits, dim=-1) # (B, 1, N, K)
            
            # 3. 加权求和
            out = (trans_feat * attn_weights).sum(dim=-1)
            return out

        # ==========================================================
        # 模式 C: 完整版本 (Full) - "剪枝 + 重归一化 + Sum"
        # ==========================================================
        elif self.mode == 'full':
            # 1. 计算 Mask (剪枝)
            gate_logits = self.gate_net(x)
            mask = gumbel_sigmoid(gate_logits, tau=1.0, hard=True,training=self.training) # (B, 1, N, K)
            
            # 2. 计算 Logits (打分)
            attn_logits = self.attn_net(x)
            
            # 3. Masked Softmax (只在保留下来的邻居中归一化)
            attn_logits = attn_logits.masked_fill(mask < 0.5, -1e9)
            attn_weights = F.softmax(attn_logits, dim=-1)
            
            # 4. 加权求和
            out = (trans_feat * attn_weights).sum(dim=-1)
            return out
            
        else:
            raise ValueError("Unknown mode")
class DGCNNpp(nn.Module):
    def __init__(self, opt,mode='full',block_type='gsl'):
        super(DGCNNpp, self).__init__()
        self.opt = opt
        self.k = opt.k  # 建议: 将 k 设大一点 (如 40)，利用剪枝机制
        self.mode = mode
        if block_type == 'gsl':
            self.GSL_Block = GSL_Block
        elif block_type == 'soft':
            self.GSL_Block = Soft_GSL_Block
        elif block_type == 'soft_pe':
            self.GSL_Block = Soft_GSL_Block_PE
        elif block_type == 'grouped_soft_pe':
            self.GSL_Block = Grouped_Soft_GSL_Block_PE
        else:
            raise ValueError("Unknown block type")
        channels = 64
        
        # --- Block 1 定义 ---
        # 原始输入特征假设为 9 (args.in_channels)，get_graph_feature 后变为 18
        in_dim_1 = 9
        
        # 定义原始的特征变换层 (保持参数量级一致)
        trans_layer_1 = nn.Sequential(
            nn.Conv2d(in_dim_1 * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # 封装进 GSL Block
        self.gsl_block1 = self.GSL_Block(opt,in_dim_1, channels, self.k, trans_layer_1,mode = self.mode)


        # --- Block 2 定义 ---
        # 输入是上一层的 channels (64)，构图后是 128
        in_dim_2 = channels
        
        trans_layer_2 = nn.Sequential(
            nn.Conv2d(in_dim_2 * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.gsl_block2 = self.GSL_Block(opt,in_dim_2, channels, self.k, trans_layer_2,mode=self.mode)


        # --- Block 3 定义 ---
        in_dim_3 = channels
        
        trans_layer_3 = nn.Sequential(
            nn.Conv2d(in_dim_3 * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.gsl_block3 = self.GSL_Block(opt,in_dim_3, channels, self.k, trans_layer_3,mode=self.mode)


        # --- 后处理保持不变 ---
        self.fusion_block = nn.Sequential(nn.Conv2d(channels * 3, 1024, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(1024),
                                          nn.LeakyReLU(negative_slope=0.2))

        fusion_dims = 1024 + channels * 3
        self.prediction = nn.Sequential(
            nn.Conv1d(fusion_dims, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=opt.dropout),
            nn.Conv1d(256, 13, kernel_size=1, bias=True) # 假设13类
        )

    def forward(self, inputs):
        # inputs shape: (B, C, N)
        if inputs.dim() == 4:
            inputs = inputs.squeeze(-1)
            
        batch_size = inputs.size(0)
        num_points = inputs.size(2)
        coords = inputs[:, 0:3, :] 
        # ---------------- Block 1 ----------------
        # 1. 几何空间 KNN (Candidates)
        idx_1 = knn(inputs[:, 0:3, :], k=self.k) 
        # 2. 获取边缘特征 (B, 2*Cin, N, K)
        edge_feat_1 = get_graph_feature(inputs, k=self.k, idx=idx_1)

        edge_coords_1 = get_graph_feature(coords, k=self.k, idx=idx_1) 
        rel_pos_1 = edge_coords_1[:, 3:6, :, :] # (B, 3, N, K) -> 取 x_j - x_i 部分
        
        # 4. GSL 聚合 (传入 rel_pos)
        net_1 = self.gsl_block1(edge_feat_1, rel_pos=rel_pos_1) 

        # ---------------- Block 2 ----------------
        # 1. 动态 KNN (基于特征空间构图)
        idx_2 = knn(net_1, k=self.k) 
        
        # 2. 获取边缘特征 (B, 128, N, K) - 这里的特征是抽象的
        edge_feat_2 = get_graph_feature(net_1, k=self.k, idx=idx_2)
        
        # 3. [关键修正] 获取当前图结构下的"几何"相对位置
        # 我们使用 feature 算出的 idx_2，去原始 coords 里找邻居，计算几何距离
        edge_coords_2 = get_graph_feature(coords, k=self.k, idx=idx_2)
        rel_pos_2 = edge_coords_2[:, 3:6, :, :] # (B, 3, N, K)
        
        # 4. GSL 聚合
        net_2 = self.gsl_block2(edge_feat_2, rel_pos=rel_pos_2) 

        # ---------------- Block 3 ----------------
        # 1. 动态 KNN
        idx_3 = knn(net_2, k=self.k)
        
        # 2. 特征图
        edge_feat_3 = get_graph_feature(net_2, k=self.k, idx=idx_3)
        
        # 3. [关键修正] 几何相对位置
        edge_coords_3 = get_graph_feature(coords, k=self.k, idx=idx_3)
        rel_pos_3 = edge_coords_3[:, 3:6, :, :]
        
        # 4. GSL 聚合
        net_3 = self.gsl_block3(edge_feat_3, rel_pos=rel_pos_3) 


        # ---------------- Fusion & Prediction (不变) ----------------
        feats = torch.cat((net_1, net_2, net_3), dim=1) # (B, 192, N)

        fusion = self.fusion_block(feats.unsqueeze(-1)) 
        fusion = F.adaptive_max_pool2d(fusion, (1, 1))
        fusion = fusion.repeat(1, 1, num_points, 1).squeeze(-1)
        
        out = torch.cat((fusion, feats), dim=1)
        return self.prediction(out)

class GumbelScheduler:
    def __init__(self, total_epochs, start_tau=5.0, end_tau=0.1, switch_hard_epoch=20):
        self.total_epochs = total_epochs
        self.start_tau = start_tau
        self.end_tau = end_tau
        self.switch_hard_epoch = switch_hard_epoch # 在第几个epoch切换为hard模式
        
    def get_tau(self, epoch):
        # 简单的指数衰减或线性衰减
        # 这里使用指数衰减: tau = start * (end/start)^(epoch/total)
        tau = self.start_tau * (self.end_tau / self.start_tau) ** (epoch / self.total_epochs)
        return max(tau, self.end_tau)

    def get_hard(self, epoch):
        # 初期用 Soft 模式训练，让梯度平滑；后期用 Hard 模式逼近离散结构
        return epoch >= self.switch_hard_epoch

class Robust_GSL_Block(nn.Module):
    def __init__(self,opt, in_channels, out_channels, k, transform_layers, 
                 mode='full', min_retention_ratio=0.2):
        """
        Args:
            mode: 'full' (完整), 'gating_only' (仅剪枝), 'attn_only' (仅注意力)
            min_retention_ratio: 强制保留邻居的最低比例 (防止塌缩)
        """
        super(Robust_GSL_Block, self).__init__()
        self.k = k
        self.mode = mode
        self.min_k = int(k * min_retention_ratio)
        self.trans = transform_layers

        # ============================================================
        # 组件初始化 (按需加载，节省参数)
        # ============================================================
        
        # 1. 门控网络 (用于 full 和 gating_only)
        if self.mode in ['full', 'gating_only']:
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                nn.Conv2d(8, 1, kernel_size=1, bias=True)
            )
            # [稳定性技巧 1] Bias 正值初始化: 初期保留大部分边
            # nn.init.constant_(self.gate_net[-1].bias, 2.0)

        # 2. 注意力网络 (用于 full 和 attn_only)
        if self.mode in ['full', 'attn_only']:
            self.attn_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                nn.Conv2d(8, 1, kernel_size=1, bias=True)
            )

        # 3. 混合聚合系数 Alpha (仅用于 full)
        if self.mode == 'full':
            # 初始设为 0.5 (Sigmoid后)，让网络自己学
            self.agg_alpha = nn.Parameter(torch.tensor(0.0)) 

    def gumbel_sigmoid_sample(self, logits, tau, hard):
        if not self.training:
            return (logits >= 0).float()
            k_retain = int(logits.size(-1) * 0.5)  # 保留一半邻居
            # 找第 k_retain 大的值作为动态阈值
            topk_val, _ = torch.topk(logits, k=k_retain, dim=-1)
            threshold = topk_val[:, :, :, -1].unsqueeze(-1)
            
            # 生成硬 Mask
            y_hard = (logits >= threshold).float()
            return y_hard

        """ Gumbel-Sigmoid 采样辅助函数 """
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = torch.sigmoid(gumbels)

        if hard:
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def forward(self, x, tau=1.0, hard=False):
        """
        Returns:
            out: 聚合后的特征
            gate_logits: 用于计算稀疏 Loss (如果是 attn_only 模式则为 None)
        """
        # 0. 基础特征变换 (B, Cout, N, K)
        trans_feat = self.trans(x)

        # ============================================================
        # 分支 A: 仅注意力 (Attention Only) -> 类似 Point Transformer
        # ============================================================
        if self.mode == 'attn_only':
            attn_logits = self.attn_net(x)
            
            # [稳定性技巧 2] 数值稳定化 (减最大值)
            attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]
            
            # 标准 Softmax
            weights = F.softmax(attn_logits, dim=-1)
            
            # 加权求和
            out = (trans_feat * weights).sum(dim=-1)
            return out, None # 无 Logits，不计算稀疏 Loss

        # ============================================================
        # 分支 B & C: 包含门控 (Gating Only / Full)
        # ============================================================
        
        # 1. 计算门控 Logits
        gate_logits = self.gate_net(x) # (B, 1, N, K)

        # [稳定性技巧 3] 保底机制 (Safety Net)
        # 找出 Logits 最大的 min_k 个值的阈值
        topk_val, _ = torch.topk(gate_logits, k=self.min_k, dim=-1)
        min_threshold = topk_val[:, :, :, -1].unsqueeze(-1)
        force_keep_mask = (gate_logits >= min_threshold).float()

        # 2. Gumbel 采样
        mask = self.gumbel_sigmoid_sample(gate_logits, tau, hard)

        # 在 Hard 模式下，强制合并保底 Mask
        if hard:
            mask = torch.max(mask, force_keep_mask)

        # --- 分支 B: 仅门控 (Gating Only) ---
        if self.mode == 'gating_only':
            # 类似 DGCNN，但剔除 mask=0 的邻居
            # 使用 -inf 填充，不影响 Max Pooling
            masked_trans = trans_feat.masked_fill(mask < 0.5, -1e9)
            out = masked_trans.max(dim=-1)[0]
            return out, gate_logits

        # --- 分支 C: 完整模式 (Full) ---
        if self.mode == 'full':
            # 1. 计算注意力 Logits
            attn_logits = self.attn_net(x)
            attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]
            
            # 2. Masked Attention
            # 将被剪枝的边 Attention Score 设为负无穷
            paddings = torch.ones_like(attn_logits) * (-1e9)
            
            # Soft 模式下如果 mask 是连续值，怎么处理？
            # 策略：Hard 阶段才做彻底截断；Soft 阶段全关注，但 mask 会乘在最后
            if hard:
                masked_attn_logits = torch.where(mask > 0.5, attn_logits, paddings)
            else:
                masked_attn_logits = attn_logits 

            weights = F.softmax(masked_attn_logits, dim=-1)

            # Soft 模式下的额外门控乘法
            if not hard:
                weights = weights * mask
                # 重新归一化 (防止除0)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
            
            # 3. 混合聚合 (Mixed Aggregation)
            # 路径 1: 加权和
            out_sum = (trans_feat * weights).sum(dim=-1)
            
            # 路径 2: Max Pooling (仅在保留的边里)
            if hard:
                masked_trans = trans_feat.masked_fill(mask < 0.5, -1e9)
            else:
                masked_trans = trans_feat
            out_max = masked_trans.max(dim=-1)[0]
            
            # 融合
            alpha = torch.sigmoid(self.agg_alpha)
            out = alpha * out_sum + (1 - alpha) * out_max
            
            return out, gate_logits

class DGCNNpp_Robust(nn.Module):
    def __init__(self, opt, mode='full'):
        super(DGCNNpp_Robust, self).__init__()
        self.opt = opt
        self.k = opt.k
        self.mode = mode
        channels = 64
        
        # 定义变换层 (trans_layer_1, 2, 3...) 代码略，同前
        # 关键是实例化 Block
        in_dim_1 = 9

        trans_layer_1 = nn.Sequential(
            nn.Conv2d(in_dim_1 * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Block 1
        self.block1 = Robust_GSL_Block(opt,
           in_dim_1, channels, self.k, trans_layer_1, 
            mode=self.mode, min_retention_ratio=0.25
        )
        
        # Block 2
        in_dim_2 = channels
        
        trans_layer_2 = nn.Sequential(
            nn.Conv2d(in_dim_2 * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.block2 = Robust_GSL_Block(opt,
            in_dim_2, channels, self.k, trans_layer_2, 
            mode=self.mode, min_retention_ratio=0.25
        )
        
        # ... Block 3, Fusion, Prediction 同前 ...
        in_dim_3 = channels
        
        trans_layer_3 = nn.Sequential(
            nn.Conv2d(in_dim_3 * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.block3 = Robust_GSL_Block(opt,in_dim_3, channels, self.k, trans_layer_3,mode=self.mode, min_retention_ratio=0.25)


        # --- 后处理保持不变 ---
        self.fusion_block = nn.Sequential(nn.Conv2d(channels * 3, 1024, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(1024),
                                          nn.LeakyReLU(negative_slope=0.2))

        fusion_dims = 1024 + channels * 3
        self.prediction = nn.Sequential(
            nn.Conv1d(fusion_dims, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=opt.dropout),
            nn.Conv1d(256, 13, kernel_size=1, bias=True) # 假设13类
        )

    def forward(self, inputs, tau=1.0, hard=False):
        # ... KNN 和 get_graph_feature 代码略 ...
        
        logits_list = []
        
        # Block 1
        idx = knn(inputs[:, 0:3, :], k=self.k)
        edge_feat1 = get_graph_feature(inputs, k=self.k, idx=idx)
        out1, logit1 = self.block1(edge_feat1, tau, hard)
        if logit1 is not None: logits_list.append(logit1)
        
        # Block 2
        # ... 略 ...
        edge_feat2 = get_graph_feature(out1, k=self.k)
        out2, logit2 = self.block2(edge_feat2, tau, hard)
        if logit2 is not None: logits_list.append(logit2)
        
        # ... Block 3 & Fusion ...
        edge_feat3 = get_graph_feature(out2, k=self.k)
        out3, logit3 = self.block3(edge_feat3,tau,hard)
        if logit3 is not None: logits_list.append(logit3)
        feats = torch.cat((out1, out2, out3), dim=1)
        fusion = self.fusion_block(feats.unsqueeze(-1))
        fusion = F.adaptive_max_pool2d(fusion, (1, 1))
        fusion = fusion.repeat(1, 1, inputs.size(2), 1).squeeze(-1)
        out = torch.cat((fusion, feats), dim=1)
        pred = self.prediction(out)
        
        # 返回预测结果和 Logits 列表 (用于 Loss 计算)
        return pred, logits_list


class Soft_GSL_Block(nn.Module):
    def __init__(self,opt, in_channels, out_channels, k, transform_layers, mode='full'):
        """
        Args:
            in_channels: 输入特征通道数 (用于计算 Gate 和 Attn 的输入)
            out_channels: 变换后的特征通道数 (Gate 的输出维度必须和它匹配)
            k: 邻居数
            transform_layers: 原始的特征变换层 (MLP)
            mode: 
                - 'gating_only': 通道软门控 + Max Pooling (去噪流)
                - 'attn_only':   空间注意力 + Sum Pooling (聚合流)
                - 'full':        通道软门控 + (注意力Sum + Max) (混合流)
        """
        super(Soft_GSL_Block, self).__init__()
        self.k = k
        self.mode = mode
        self.trans = transform_layers
        self.out_channels = out_channels

        # ------------------------------------------------------------------
        # 组件 A: 通道门控网络 (Channel-wise Gating) - 仅在 gating_only 或 full 模式下初始化
        # 作用: 给每个特征通道打分 (0~1)，抑制噪声
        # ------------------------------------------------------------------
        if self.mode in ['gating_only', 'full']:
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                nn.Conv2d(8, 1, kernel_size=1, bias=True),
                nn.Sigmoid() # <--- 关键：Soft Gating 使用 Sigmoid
            )
            # 初始化技巧：Bias 设为正值，让初始 Gate 接近 1 (不阻拦特征)，保证初期训练稳定
            nn.init.constant_(self.gate_net[-2].bias, 2.0)

        # ------------------------------------------------------------------
        # 组件 B: 空间注意力网络 (Spatial Attention) - 仅在 attn_only 或 full 模式下初始化
        # 作用: 给每个邻居打分，归一化后加权
        # ------------------------------------------------------------------
        if self.mode in ['attn_only', 'full']:
            self.attn_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                nn.Conv2d(8, 1, kernel_size=1, bias=True)
            )

    def forward(self, x, rel_pos=None):
        # x: (B, 2*Cin, N, K) -> 原始输入特征
        
        # 1. 基础特征变换 (所有模式通用)
        # trans_feat: (B, Cout, N, K)
        trans_feat = self.trans(x)

        # ==========================================================
        # 模式 A: 仅通道软门控 (Gating Only)
        # ==========================================================
        if self.mode == 'gating_only':
            # 1. 计算通道权重 (B, Cout, N, K)
            gate_map = self.gate_net(x)
            
            # 2. 软去噪 (Soft Scaling)
            clean_feat = trans_feat * gate_map
            
            # 3. 聚合 (依然使用 Max Pooling，因为我们只是想“净化”特征)
            out = clean_feat.max(dim=-1)[0]
            return out

        # ==========================================================
        # 模式 B: 仅空间注意力 (Attention Only)
        # ==========================================================
        elif self.mode == 'attn_only':
            # 1. 计算空间分数 (B, 1, N, K)
            attn_logits = self.attn_net(x)
            
            # 2. Softmax 归一化
            attn_weights = F.softmax(attn_logits, dim=-1)
            
            # 3. 加权求和
            out = (trans_feat * attn_weights).sum(dim=-1)
            return out

        # ==========================================================
        # 模式 C: 完整混合模式 (Full: Filter-then-Focus)
        # ==========================================================
        elif self.mode == 'full':
            # 1. 先去噪 (Gating)
            gate_map = self.gate_net(x)
            clean_feat = trans_feat * gate_map # 这里的特征已经被“清洗”过，幅度被调整
            
            # 2. 再聚焦 (Attention)
            attn_logits = self.attn_net(x)
            attn_weights = F.softmax(attn_logits, dim=-1)
            
            # 3. 双路聚合
            # 路一：利用 Attention 做精细聚合
            out_attn = (clean_feat * attn_weights).sum(dim=-1)
            
            # 路二：利用 Max Pooling 提取显著特征 (作为保底残差)
            # 注意：这里的 Max 是在 clean_feat 上做的，所以噪声已经被抑制了，Max 效果会比 Baseline 好
            out_max = clean_feat.max(dim=-1)[0]
            
            # 4. 融合
            out = out_attn + out_max
            return out
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class Soft_GSL_Block_PE(nn.Module):
    def __init__(self, opt, in_channels, out_channels, k, transform_layers, mode='full'):
        """
        Args:
            in_channels: 原始点云特征维度 (假设前3维是XYZ坐标)
            out_channels: 变换后的特征通道数 (必须与 transform_layers 的输出一致)
            k: 邻居数
            transform_layers: 原始的特征变换层 (MLP)
            mode: 'gating_only' / 'attn_only' / 'full'
        """
        super(Soft_GSL_Block_PE, self).__init__()
        self.k = k
        self.mode = mode
        self.trans = transform_layers
        self.in_channels = in_channels 
        self.out_channels = out_channels

        # ==================================================================
        # [新增] 组件 P: 位置编码网络 (Positional Encoding Net)
        # 作用: 将相对坐标 (3维) 映射为 高维位置特征 (out_channels)
        # ==================================================================
        self.pos_net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        )

        # ==================================================================
        # 组件 A: 通道门控网络 (保持标量门控，负责结构去噪)
        # ==================================================================
        if self.mode in ['gating_only', 'full']:
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 16, kernel_size=1, bias=False), # 稍微增加隐层
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                nn.Conv2d(16, 1, kernel_size=1, bias=True), # 输出 1 (Scalar Gate)
                nn.Sigmoid() 
            )
            nn.init.constant_(self.gate_net[-2].bias, 2.0)

        # ==================================================================
        # 组件 B: 向量注意力网络 (升级为 Vector Attention)
        # 作用: 输出维度变为 out_channels，实现通道级加权
        # ==================================================================
        if self.mode in ['attn_only', 'full']:
            self.attn_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels // 4, kernel_size=1, bias=False), # 瓶颈层
                nn.BatchNorm2d(out_channels // 4),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                # [关键改动] 输出 out_channels (Vector), 而不是 1
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=True) 
            )

    def forward(self, x, rel_pos=None):
        """
        x: (B, 2*Cin, N, K) 
           假设输入的结构是 cat(x_i, x_j - x_i)。
           并且原始特征的前3维是 XYZ 坐标。
           那么 x[:, in_channels : in_channels+3, ...] 就是 p_j - p_i (相对坐标)。
        """
        
        # ------------------------------------------------------------------
        # Step 1: 提取位置编码 (Positional Encoding)
        # ------------------------------------------------------------------
        # 提取相对坐标 delta_p (B, 3, N, K)
        # 注意：这里依赖于输入数据格式，确保前3通道是坐标
        if rel_pos is None:
            rel_pos = x[:, self.in_channels : self.in_channels + 3, :, :]
        
        # 计算位置特征 delta (B, Cout, N, K)
        pos_enc = self.pos_net(rel_pos)

        # ------------------------------------------------------------------
        # Step 2: 基础特征变换 + 位置注入 (Value Injection)
        # ------------------------------------------------------------------
        # trans_feat: (B, Cout, N, K)
        # Point Transformer 核心公式: Value = MLP(Features) + Delta
        trans_feat = self.trans(x) + pos_enc 

        # ==========================================================
        # 模式 A: 仅通道软门控 (Gating Only)
        # ==========================================================
        if self.mode == 'gating_only':
            # Gate 依然只看特征 (或可选择也把 pos 加进去，这里暂只用原始特征)
            gate_map = self.gate_net(x) # (B, 1, N, K)
            
            # 软去噪 (广播乘法)
            clean_feat = trans_feat * gate_map 
            
            # 聚合
            out = clean_feat.max(dim=-1)[0]
            return out

        # ==========================================================
        # 模式 B: 仅向量注意力 (Vector Attention Only)
        # ==========================================================
        elif self.mode == 'attn_only':
            # 1. 计算原始 Attention Logits (B, Cout, N, K)
            attn_logits = self.attn_net(x)
            
            # 2. 位置注入 (Attention Injection)
            # Point Transformer 核心公式: Logits = MLP(Features) + Delta
            attn_logits = attn_logits + pos_enc
            
            # 3. 向量 Softmax (沿 K 维度)
            attn_weights = F.softmax(attn_logits, dim=-1)
            
            # 4. 加权求和 (Hadamard Product -> Sum)
            # (B, C, N, K) * (B, C, N, K) -> Sum
            out = (trans_feat * attn_weights).sum(dim=-1)
            return out

        # ==========================================================
        # 模式 C: 完整混合模式 (Full: PE + Gate + Vector Attn)
        # ==========================================================
        elif self.mode == 'full':
            # --- 分支 1: 门控去噪 ---
            gate_map = self.gate_net(x) # (B, 1, N, K)
            # 使用门控清洗带有位置信息的特征
            clean_feat = trans_feat * gate_map 
            
            # --- 分支 2: 向量注意力 ---
            attn_logits = self.attn_net(x)
            # 位置注入到 Attention
            attn_logits = attn_logits + pos_enc
            
            # 关键：我们希望被 Gate 抑制的区域，Attention 也不要关注
            # 软结合：将 Gate (0~1) 转换为 Log Space 的 Bias 加进去
            # 当 Gate->0, Log(Gate)-> -inf, Softmax->0
            # attn_logits = attn_logits + torch.log(gate_map + 1e-6)
            
            # 计算向量权重
            attn_weights = F.softmax(attn_logits, dim=-1) # (B, Cout, N, K)
            
            # --- 聚合 ---
            # 路一: Vector Attention 聚合
            out_attn = (trans_feat * attn_weights).sum(dim=-1)
            
            # 路二: Max Pooling 保底 (基于清洗后的特征)
            out_max = clean_feat.max(dim=-1)[0]
            
            # 融合
            out = out_attn + out_max
            return out
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
class Grouped_Soft_GSL_Block_PE(nn.Module):
    def __init__(self, opt, in_channels, out_channels, k, transform_layers, mode='full', groups=8):
        """
        groups: 将特征分为多少组进行注意力加权 (建议 4, 8 或 16)
        """
        super(Grouped_Soft_GSL_Block_PE, self).__init__()
        self.k = k
        self.mode = mode
        self.trans = transform_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        
        # 确保通道数能被整除
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        self.channels_per_group = out_channels // groups

        # ------------------------------------------------------------------
        # P. 位置编码网络 (Positional Encoding)
        # ------------------------------------------------------------------
        self.pos_net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        )
        
        # [新增] PE 投影层: 将 PE 从 out_channels 降维到 groups
        # 用于将位置信息注入到分组注意力中
        self.pe_proj = nn.Conv2d(out_channels, groups, kernel_size=1, bias=True)

        # ------------------------------------------------------------------
        # A. 门控网络 (Gate) - 保持标量，控制结构
        # ------------------------------------------------------------------
        if self.mode in ['gating_only', 'full']:
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 16, kernel_size=1, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                nn.Conv2d(16, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            nn.init.constant_(self.gate_net[-2].bias, 2.0)

        # ------------------------------------------------------------------
        # B. 分组注意力网络 (Grouped Attention)
        # ------------------------------------------------------------------
        if self.mode in ['attn_only', 'full']:
            self.attn_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels // 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.LeakyReLU(0.2),
                nn.Dropout(opt.dropout),
                # [关键修改] 输出通道数为 groups (例如 8)，而不是 out_channels (64)
                nn.Conv2d(out_channels // 4, groups, kernel_size=1, bias=True)
            )

    def forward(self, x, rel_pos=None):
        # x: (B, 2*Cin, N, K)
        
        # 1. 提取位置编码
        if rel_pos is None:
             rel_pos = x[:, self.in_channels : self.in_channels + 3, :, :]
        pos_enc = self.pos_net(rel_pos) # (B, Cout, N, K)

        # 2. 特征变换 + 位置注入 (Value Injection)
        # 特征流依然保持 Full Channel，每个通道都有独立的位置信息
        trans_feat = self.trans(x) + pos_enc 

        # ========================= 模式分支 =========================
        
        # 模式 A: 仅门控 (Gate 不受 Group 影响)
        if self.mode == 'gating_only':
            gate_map = self.gate_net(x)
            out = (trans_feat * gate_map).max(dim=-1)[0]
            return out

        # 模式 B/C: 需要计算分组注意力
        elif self.mode in ['attn_only', 'full']:
            
            # --- 1. 计算 Group Logits (B, Groups, N, K) ---
            attn_logits = self.attn_net(x)
            
            # --- 2. 注入位置编码 (Attention Injection) ---
            # 先将 C 维的 pos_enc 投影到 G 维
            pos_enc_grouped = self.pe_proj(pos_enc) 
            attn_logits = attn_logits + pos_enc_grouped

            # --- 3. 门控交互 (仅 Full 模式) ---
            if self.mode == 'full':
                gate_map = self.gate_net(x) # (B, 1, N, K)
                # Gate 是标量，广播加到 Group Logits 上
                attn_logits = attn_logits + torch.log(gate_map + 1e-6)
            
            # --- 4. Softmax (Group 级别) ---
            # (B, Groups, N, K)
            attn_weights_grouped = F.softmax(attn_logits, dim=-1)
            
            # --- 5. [核心步骤] 权重扩展 (Broadcasting) ---
            # 我们有 8 个权重，需要扩展回 64 个通道。
            # 假设 groups=8, out_channels=64, 那么 channels_per_group=8
            # 我们需要把每个权重重复 8 次
            
            # repeat_interleave: 在 dim=1 上，每个元素重复 8 次
            # (B, 8, N, K) -> (B, 64, N, K)
            attn_weights_expanded = attn_weights_grouped.repeat_interleave(
                self.channels_per_group, dim=1
            )

            # --- 6. 聚合 ---
            # (B, 64, N, K) * (B, 64, N, K)
            out_attn = (trans_feat * attn_weights_expanded).sum(dim=-1)

            if self.mode == 'attn_only':
                return out_attn
            
            # Full 模式: 加上 Max Pooling 保底
            if self.mode == 'full':
                clean_feat = trans_feat * gate_map
                out_max = clean_feat.max(dim=-1)[0]
                return out_attn + out_max
                
        else:
            raise ValueError(f"Unknown mode: {self.mode}")