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
    

def gumbel_sigmoid(logits, tau=1.0, hard=False):
    """
    可微的二值采样：
    - Forward: 输出近似 0 或 1 的值 (如果 hard=True，则严格输出 0/1)
    - Backward: 梯度可传导回 logits
    """
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

class Ablation_EdgeConv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, k, transform_layers, mode='full'):
        """
        mode: 
          - 'gating_only': 只有硬门控剪枝 (Gumbel-Sigmoid + Max Pooling) -> 类似 DGCNN 但能删邻居
          - 'attn_only':   只有软注意力 (Softmax + Weighted Sum) -> 类似 Point Transformer
          - 'full':        既有剪枝又有注意力 (Gumbel + Masked Softmax + Sum) -> 你的完整方法
        """
        super(Ablation_EdgeConv_Block, self).__init__()
        self.k = k
        self.mode = mode
        self.trans = transform_layers

        # 1. 定义网络组件 (根据模式按需定义，节省显存)
        
        # [Gating 组件]: 用于 'gating_only' 和 'full'
        if self.mode in ['gating_only', 'full']:
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(8, 1, kernel_size=1, bias=True)
            )
            # nn.init.constant_(self.gate_net[-1].bias, 5.0) 

        # [Attention 组件]: 用于 'attn_only' 和 'full'
        if self.mode in ['attn_only', 'full']:
            self.attn_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(8, 1, kernel_size=1, bias=True)
            )

    def forward(self, x):
        # x: (B, 2*Cin, N, K) 边缘特征
        
        # 先进行通用的特征变换 (MLP)
        trans_feat = self.trans(x) # (B, Cout, N, K)

        # ==========================================================
        # 模式 A: 仅门控 (Gating Only) - "硬剪枝 + Max Pooling"
        # ==========================================================
        if self.mode == 'gating_only':
            # 1. 计算 Mask (0 或 1)
            gate_logits = self.gate_net(x)
            mask = gumbel_sigmoid(gate_logits, tau=1.0, hard=True) # (B, 1, N, K)
            
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
            weights = F.softmax(attn_logits, dim=-1) # (B, 1, N, K)
            
            # 3. 加权求和
            out = (trans_feat * weights).sum(dim=-1)
            return out

        # ==========================================================
        # 模式 C: 完整版本 (Full) - "剪枝 + 重归一化 + Sum"
        # ==========================================================
        elif self.mode == 'full':
            # 1. 计算 Mask (剪枝)
            gate_logits = self.gate_net(x)
            mask = gumbel_sigmoid(gate_logits, tau=1.0, hard=True)
            
            # 2. 计算 Logits (打分)
            attn_logits = self.attn_net(x)
            
            # 3. Masked Softmax (只在保留下来的邻居中归一化)
            attn_logits = attn_logits.masked_fill(mask < 0.5, -1e9)
            weights = F.softmax(attn_logits, dim=-1)
            
            # 4. 加权求和
            out = (trans_feat * weights).sum(dim=-1)
            return out
            
        else:
            raise ValueError("Unknown mode")
class DGCNNpp(nn.Module):
    def __init__(self, opt,mode='full'):
        super(DGCNNpp, self).__init__()
        self.opt = opt
        self.k = opt.k  # 建议: 将 k 设大一点 (如 40)，利用剪枝机制
        self.mode = mode
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
        self.gsl_block1 = Ablation_EdgeConv_Block(in_dim_1, channels, self.k, trans_layer_1,mode = self.mode)


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
        self.gsl_block2 = Ablation_EdgeConv_Block(in_dim_2, channels, self.k, trans_layer_2,mode=self.mode)


        # --- Block 3 定义 ---
        in_dim_3 = channels
        
        trans_layer_3 = nn.Sequential(
            nn.Conv2d(in_dim_3 * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.gsl_block3 = Ablation_EdgeConv_Block(in_dim_3, channels, self.k, trans_layer_3,mode=self.mode)


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

        # ---------------- Block 1 ----------------
        # 1. 几何空间 KNN (Candidates)
        idx = knn(inputs[:, 0:3, :], k=self.k) 
        # 2. 获取边缘特征 (B, 2*Cin, N, K)
        edge_feat_1 = get_graph_feature(inputs, k=self.k, idx=idx)
        # 3. GSL 聚合 (Gate -> Prune -> Attn -> Sum)
        net_1 = self.gsl_block1(edge_feat_1) # (B, 64, N)

        # ---------------- Block 2 ----------------
        # 动态 KNN (基于 net_1 特征空间)
        edge_feat_2 = get_graph_feature(net_1, k=self.k) 
        net_2 = self.gsl_block2(edge_feat_2) # (B, 64, N)

        # ---------------- Block 3 ----------------
        edge_feat_3 = get_graph_feature(net_2, k=self.k)
        net_3 = self.gsl_block3(edge_feat_3) # (B, 64, N)

        # ---------------- Fusion & Prediction (不变) ----------------
        feats = torch.cat((net_1, net_2, net_3), dim=1) # (B, 192, N)

        fusion = self.fusion_block(feats.unsqueeze(-1)) 
        fusion = F.adaptive_max_pool2d(fusion, (1, 1))
        fusion = fusion.repeat(1, 1, num_points, 1).squeeze(-1)
        
        out = torch.cat((fusion, feats), dim=1)
        return self.prediction(out)