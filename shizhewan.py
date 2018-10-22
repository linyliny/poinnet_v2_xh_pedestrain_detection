#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:26:07 2018

@author: xh
"""

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
from torch.nn import utils as nn_utils

import PointNet
import utils
import PointCloudShow
import PointCloudShowVTK


#  data
#tracking = '0016'                                                              # tracking-th data collection
#order = 11 
#aaa, bbb = utils.loadData()
#a_1 = aaa[1:3]
#b_1 = bbb[1:3]
#a_1 = torch.Tensor(a_1)
#print(a_1)
batch_size = 64

dataset = utils.KITTI_Dateset(all_data=False)
#ee, _ = dataset[488]
#print(ee.size()[0])
len_set = len(dataset)
#print(len_set)
#max_len = 0
#ii = 0
#for i in range(0,len_set):
#    po, la = dataset[i]
#    one_len = po.size()[0]
#    print(str(i), one_len)
#    if one_len >= max_len:
#        max_len = one_len
#        ii = i
#print(str(ii), max_len)
yu = len_set % batch_size
batch_num = int((len_set - yu) / batch_size)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
#print(dataloader)
#  net
net = PointNet.PointNet()
net.cuda()

weight = np.array([1.0, 1.0])
weight = torch.from_numpy(weight)
weight = weight.float().cuda()
criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = optim.Adam(net.parameters(), lr=0.001)

#zheng = 0
#zong = 0
for i in range(0, batch_num):
    loss_num = 0
    for j in range(0, batch_size):
        po, la = dataset[i*batch_size+j]
        po = po.unsqueeze(0)
        po, la = Variable(po), Variable(la)
#        print(po)
        po = po.transpose(2, 1)
#        print(po)
        po, la = po.cuda(), la.cuda()
        pr, trans = net(po)
        loss = criterion(pr, la)
        loss_num = loss + loss_num
#    print(loss_num)
    loss_num.backward()
    optimizer.step()
    
    
#    points, target = data
#    print(points, target)
#    points, target = Variable(points), Variable(target)
#    points = points.transpose(2, 1)
#    points, target = points.cuda(), target.cuda()
#    x, trans = net(points)
#    break
#    ee = points
#    print(ee)
#    print(points.size()[2])
#    if int(target.cpu()) == 1:
#        zheng = zheng + 1
#    zong = zong + 1
#print(zong, zheng)


#import os
#import random
#
#import numpy as np
#import torch
#import torch.optim as optim
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable
#
#import utils
#import PointNet
#
#random.seed(random.randint(1, 10000))
#torch.manual_seed(random.randint(1, 10000))
#
## hyper-parameter
#batch_size = 64
#n_epoch    = 3
#
#tracking   = '0000'
#
#
#dataset = utils.KITTI_Dateset(tracking=tracking, all_data=False)
#dataset_rand = torch.utils.data.sampler.RandomSampler(dataset)
##ddd = dataset_rand[33]
#print(dataset[22],dataset_rand)
#print(len(dataset),len(dataset_rand))