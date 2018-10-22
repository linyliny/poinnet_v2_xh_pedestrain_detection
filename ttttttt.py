#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:44:40 2018

@author: xh
"""

import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils
import PointNet

random.seed(random.randint(1, 10000))
torch.manual_seed(random.randint(1, 10000))

# hyper-parameter
batch_size = 128
n_epoch    = 30

tracking   = '0000'


dataset = utils.KITTI_Dateset(tracking=tracking, all_data=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(tracking + ' contains %d grids' % (len(dataset)))
print('n_epoch = %d, batch_size = %d' % (n_epoch, batch_size))

net = PointNet.PointNet(k=2)
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)


if __name__ == '__main__':
    
    # checkpoint = utils.load_checkpoint()
    # net.load_state_dict(checkpoint['state_dict'])
    for epoch in range(n_epoch):
        correct = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        for batch_index, data in enumerate(dataloader, 0):
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            pred, trans = net(points)

            pred = pred.view(-1, 2)
            target = target.view(-1)

            weight = np.array([1.0, 1.0])
            weight = torch.from_numpy(weight)
            weight = weight.float().cuda()
            criterion = nn.CrossEntropyLoss(weight=weight)

            loss = criterion(pred, target) * 10
            norm = torch.matmul(trans, trans.transpose(2, 1))
            I = Variable(torch.eye(64).repeat(loss.size()[0], 1, 1)).cuda()
            L2_loss = torch.norm(norm - I, 2)                                  # L2 loss: norm(I - A*(AT)); A: feature alignment matrix
            loss += L2_loss

            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]                                  # set the most probable one to 1

            correct += pred_choice.eq(target.data).cpu().sum()                 # sum the right prediction

            # TP (true positive): predict = 1, label = 1
            TP += ((pred_choice == 1) & (target.data == 1)).cpu().sum()
            # TN (true negative): predict = 0, label = 0
            TN += ((pred_choice == 0) & (target.data == 0)).cpu().sum()
            # FN (false positive): predict = 0, label = 1
            FN += ((pred_choice == 0) & (target.data == 1)).cpu().sum()
            # FP (false negative): predict = 1, label = 0
            FP += ((pred_choice == 1) & (target.data == 0)).cpu().sum()

            p = TP / (TP + FP)
            r = TP / (TP + FN)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('batch_index: [%d/%d]' % (batch_index, len(dataloader)),
                  'Train epoch: [%d]' % (epoch),
                  'Loss %.4f L2_loss %.4f' % (loss.data[0], L2_loss.data[0]),
                  'acc:%.4f p:%.4f r:%.4f F1:%.4f' % (acc, p, r, F1))

        checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        utils.save_checkpoint(checkpoint, 'models/myModels.pth')
        print('model saved')

