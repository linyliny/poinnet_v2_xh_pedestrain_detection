import os
import sys
import random
import copy
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.stats import mode
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import PointNet
import utils
import PointCloudShow
import PointCloudShowVTK

tracking = '0015'                                                              # tracking-th data collection
order = 98                                                                   # order-th frame in tracking-th data collection


color_cate = [[225, 225, 225], [225,   0,   0], [  0, 225,   0], [  0,   0, 225],
              [112, 112, 112], [  0,   0, 112], [  0, 112,   0], [112,   0,   0],
              [112, 112,   0], [  0, 112, 112]]
color_cate = np.array(color_cate)

if __name__ == '__main__':
    
    # read data
    data = utils.loadOneVelodyneData(tracking=tracking, order=order)           
    data_copy_all = copy.deepcopy(data)                                        # data_copy_all: dim: (len(data), 3 (x, y, z))
    labels = utils.loadOneVelodyneLabel(order=order, tracking=tracking)        # labels: [frame, track_id, type, velodyne_3D (3 x 8), rotation_y]
    index = [i for i in range(len(labels)) if labels[i][0] == order]
    labels_frame = []                                                          
    for i in index:
        labels_frame.append(labels[i])
    
    # preprocess data; data, labels_frame: data and labels of order-th frame    
    data, category, cluster_labels, cluster_centers = utils.process(data, labels_frame)
    
    data_copy = copy.deepcopy(data)
    data_struct  = []
    label_struct = []

    # appoint clustered data to the category which most points belong to
    if len(cluster_labels) > 0:
        for k in range(np.max(cluster_labels) + 1):
            index = (cluster_labels == k)
            if np.sum(index) != 0:
                data_struct.append(data[index])
                most_frequent = mode(category[index])[0][0][0]                 # mode(): return the modal value 
                label_struct.append(most_frequent)
                
    data_struct = np.array(data_struct)                                        # dim (num_cluster, )
    label_struct = np.array(label_struct)
    datas, labels = data_struct, label_struct
    labels = labels.reshape([-1, 1])
#    print(labels)
    
    '''--- programs above is like transData(), but just for single frame ---''' 
    
    # resize number of every data in datas to n, because of the fixed length of input of CNN
#    datas_norm = []
#    for data in datas:
#        data = utils.criterion_points(data)
#        datas_norm.append(data)
#    datas_norm = np.array(datas_norm)
#    datas = np.float32(datas)
#    datas = datas_norm
    
    # 暂时只看人; 'Pedestrian' = 1
    for k in range(len(labels)):
        if labels[k][0] != 1:
            labels[k][0] = 0
#    print(labels)
    
    '''--------------------- predict by Pointnet ---------------------------'''
    net = PointNet.PointNet(k=2)
    checkpoint = utils.load_checkpoint('models/zan/myModels7.pth')            # load parameters from file
    net.load_state_dict(checkpoint['state_dict'])
    net.cuda()

    cluster_num = len(labels)
#    pre_la = labels.reshape(-1)
    pre_la = torch.LongTensor(cluster_num)
        
    for i in range(cluster_num):
        points = datas[i]
        points = np.float32(points)
        points = torch.from_numpy(points)
        points = points.unsqueeze(0)
        points = Variable(points)
        points = points.transpose(2, 1)
        points = points.cuda()
        pred, _ = net(points)
#        print(pred)
        _, pred = torch.max(pred.data, 1)
#        print(pred)
#        pred = pred.view(-1, 2)
#        pred = pred.data.max(dim = 1)[1]
        pred = pred.cpu()
        if (pred==1).numpy():
            pre_la[i] = 1
        else:
            pre_la[i] = 0

#    print(labels)
#    pred, _ = net(points)                                                  # dim: pre: (num_cluster, k (k=2)); trans: (num_cluster, 64, 64)
#    pred = pred.view(-1, 2)
#    target = target.view(-1)                                                   # dim: target: (num_cluster, 1)
#
#    weight = np.array([1.0, 1.0])
#    weight = torch.from_numpy(weight)
#    weight = weight.float().cuda()
#    criterion = nn.CrossEntropyLoss(weight=weight)
#    
#    loss = criterion(pred, target) * 10
#
#    norm = torch.matmul(trans, trans.transpose(2, 1))                          # dim: norm: (num_cluster, 64, 64)
#    I = Variable(torch.eye(64).repeat(loss.size()[0], 1, 1)).cuda()            # dim: I: (1, 64, 64)
#    L2_loss = torch.norm(norm - I, 2)
#    loss += L2_loss
#    
#    pred = pred.data.max(dim = 1)[1]                                           # predictions for each cluster; dim: (num_cluster, 1)
#    correct = pred.eq(target.data).cpu().sum()                                 # number of correct predictions (eq: compare equality element-wise)
#    print('Loss', loss.data[0],
#          'L2_loss', L2_loss.data[0],
#          'correct: %d/%d' % (correct, len(datas)))                            # correct ratio

    labels = labels.reshape(-1)
    pre_la = pre_la.numpy()
    print('pred:    ', pre_la.reshape(-1))
    print('label:   ', labels.reshape(-1))

    '''-------------------------- show -------------------------------------'''
    # data_copy_all: raw data
    category_raw_data = utils.classifyRawData(data_copy_all,labels_frame)
    N = len(data_copy_all)
    color_all = np.zeros([N, 3])
    for n in range(N):
        color_all[n] = color_cate[int(category_raw_data[n, 0])]

    # data_copy: preprocessed data
    N = len(data_copy)
    color = np.zeros([N, 3])                                                   # color: true color
    for n in range(N):
        color[n] = color_cate[category[n, 0]]                                  # different color for different category


    # 预测结果
    pred = list(pre_la)
    pre_data = []
    pre_data = np.array(pre_data)
    pre_data.shape = -1, 3
    pre_category = np.zeros(category.shape)
    if len(cluster_labels) > 0:
        for k in range(np.max(cluster_labels) + 1):
            index = (cluster_labels == k)
            if np.sum(index) != 0:
                pre_data = np.append(pre_data, data_copy[index], axis=0)
                pre_category[index] = pred.pop(0)
    N = len(pre_data)
    pre_color = np.zeros([N, 3])
    for n in range(N):
        pre_color[n] = color_cate[int(pre_category[n, 0])]
    # print(np.max(cluster_labels))
    # index = cluster_labels == 6
    # data = data[index]
    # color = color[index]
    
    print(len(cluster_centers), 'clusters')

    '''------------------------- vispy -------------------------------------'''
    # print('point number:', len(data))
    # PointCloudShow.show(data_copy_all, np.zeros([len(data_copy_all), 3]))
    # PointCloudShow.show(data_copy, color)
    # PointCloudShow.show(data_copy, pre_color)
    
    '''--------------------------- vtk -------------------------------------'''
    PointCloudShowVTK.show(data_copy_all, color_all)
    PointCloudShowVTK.show(data_copy, color)
    PointCloudShowVTK.show(data_copy, pre_color)
    