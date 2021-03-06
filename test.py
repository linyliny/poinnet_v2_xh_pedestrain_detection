import os
import sys
import random

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import PointNet
import utils

random.seed(random.randint(1, 10000))
torch.manual_seed(random.randint(1, 10000))

# hyper-parameter
batch_size = 32
n_epoch = 10
point_size = 500
tracking='0017'


dataset = utils.KITTI_Dateset(tracking=tracking)
print(tracking + 'contains %d girds' % (len(dataset)))
print('n_epoch = %d, batch_size = %d' % (n_epoch, batch_size))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


net = PointNet.PointNet(num_points=point_size, k=2)
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)


if __name__ == '__main__':
    checkpoint = utils.load_checkpoint()
    net.load_state_dict(checkpoint['state_dict'])
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
            print(pred.size(), target.size())
            break

            loss = criterion(pred, target) * 10
            norm = torch.matmul(trans, trans.transpose(2, 1))
            I = Variable(torch.eye(64).repeat(loss.size()[0], 1, 1)).cuda()
            L2_loss = torch.norm(norm - I, 2)
            loss += L2_loss

            loss.backward()
            # optimizer.step()

            pred_choice = pred.data.max(1)[1]

            correct += pred_choice.eq(target.data).cpu().sum()


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

            # if batch_index % 10 == 9:
        checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        # utils.save_checkpoint(checkpoint)
        # print('model saved')

