#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao, Pengliang Ji, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
import pickle
from torch.utils.data import Dataset
import zipfile

def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    # if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        # www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        www = 'https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        # os.system('mv %s %s' % ('indoor3d_sem_seg_hdf5_data', DATA_DIR))
        os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
            os.system('rm %s' % (zippath))


def prepare_test_data_semseg(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    print(partition)
    if partition == 'train':
        # data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    if not os.path.exists(data_dir):
        os.system('python prepare_data/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    download_S3DIS()
    prepare_test_data_semseg(partition)
    if partition == 'train':
        # data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def load_color_partseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/partseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.76, (0, 0, 0), 2)
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite("prepare_data/meta/partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column + 1 >= column_numbers[row]):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


def load_color_semseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/semseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite("prepare_data/meta/semseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break  
    

def translate_pointcloud(pointcloud):
    # 只针对前3维 (XYZ)
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    
    pointcloud[:, :3] = np.add(np.multiply(pointcloud[:, :3], xyz1), xyz2)
    return pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    # 只针对前3维 (XYZ) 增加噪声
    N, C = pointcloud.shape
    pointcloud[:, :3] += np.clip(sigma * np.random.randn(N, 3), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    # 修改为绕 Z 轴旋转 (旋转 X 和 Y)，这是 S3DIS 的标准做法
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    # 旋转 X(0) 和 Y(1)
    pointcloud[:,[0,1]] = pointcloud[:,[0,1]].dot(rotation_matrix) 
    return pointcloud

def interpolate_data(pointcloud, num_points):
    """ 处理点数不足的情况：随机重复采样 """
    n_points = pointcloud.shape[0]
    if n_points >= num_points:
        indices = np.random.choice(n_points, num_points, replace=False)
    else:
        indices = np.random.choice(n_points, num_points, replace=True)
    return indices

def normalize_pointcloud(pointcloud):
    """ 
    模仿 PyG 的 T.NormalizeScale(): 
    1. 中心化 (Zero-mean)
    2. 缩放到单位球内
    """
    # 假设前3维是 XYZ
    coords = pointcloud[:, :3]
    centroid = np.mean(coords, axis=0)
    coords = coords - centroid # 中心化
    m = np.max(np.sqrt(np.sum(coords**2, axis=1))) # 计算最大距离
    if m > 1e-6:
        coords = coords / m # 缩放
    pointcloud[:, :3] = coords
    return pointcloud

class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition    
        self.semseg_colors = load_color_semseg()

    def __getitem__(self, item):
        # 1. 获取完整原始数据
        pointcloud = self.data[item] 
        seg = self.seg[item]

        # 2. 改进采样逻辑 (借鉴 Code2/PyG 的动态性)
        # 不再固定取 [:4096]，而是随机抽取 4096 个点，确保每个 Epoch 看到不同的点组合
        indices = interpolate_data(pointcloud, self.num_points)
        pointcloud = pointcloud[indices]
        seg = seg[indices]

        # 3. 核心改进：坐标归一化 (关键：解决 Code1 训练曲线不平滑问题)
        # 这是 Code2 效果好的核心原因之一
        
        # pointcloud = normalize_pointcloud(pointcloud)
        # 4. 在线数据增强 (仅在训练集开启)
        if self.partition == 'train':
            # 随机打乱点序 (原有逻辑保留)
            indices = np.arange(pointcloud.shape[0])
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
            
            # 加入几何变换增强 (解决 mIoU 低的问题)
            # pointcloud = rotate_pointcloud(pointcloud)
            # pointcloud = translate_pointcloud(pointcloud)
            # pointcloud = jitter_pointcloud(pointcloud)

        
        # 5. 保持输出格式兼容原有 Workflow
        seg = torch.LongTensor(seg)
        # 确保 pointcloud 是 FloatTensor, 原有代码会在 main 中 permute
        return pointcloud.astype('float32'), seg

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = S3DIS(4096)
    test = S3DIS(4096, 'test')
    data, seg = train[0]
    print(data.shape)
    print(seg.shape)