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
batch_size = 64
n_epoch    = 50

tracking   = '0000'


dataset = utils.KITTI_Dateset(tracking=tracking, all_data=True)
dataset_rand = torch.utils.data.sampler.RandomSampler(dataset)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
len_set = len(dataset)
yu = len_set % batch_size
batch_num = int((len_set - yu) / batch_size)

print(tracking + ' contains %d grids' % (len(dataset)))
print('n_epoch = %d, batch_size = %d' % (n_epoch, batch_size))

net = PointNet.PointNet(k=2)
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00001)
loss_av = 0
loss_num_l2_av = 0

if __name__ == '__main__':
    
    checkpoint = utils.load_checkpoint('models/zan/myModels2.pth')
    net.load_state_dict(checkpoint['state_dict'])
    for epoch in range(n_epoch):
#        correct = 0
#        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(0, batch_num):
            loss_num = 0
            loss_num_l2 = 0
            optimizer.zero_grad()
            for j in range(0, batch_size):
                po, la = dataset[i*batch_size+j]
                po = po.unsqueeze(0)
                po, la = Variable(po), Variable(la)
                po = po.transpose(2, 1)
                po, la = po.cuda(), la.cuda()
                pr, trans = net(po)
                pr = pr.view(-1, 2)
                la = la.view(-1)
#                weight = np.array([1.0, 1.0])
#                weight = torch.from_numpy(weight)
#                weight = weight.float().cuda()
                criterion = nn.CrossEntropyLoss()
                loss_o = criterion(pr, la) * 10
                norm = torch.matmul(trans, trans.transpose(2, 1))
                I = Variable(torch.eye(64).repeat(loss_o.size()[0], 1, 1)).cuda()
                L2_loss = torch.norm(norm - I, 2)
                loss_o = loss_o + L2_loss * 1000
                loss_o.backward()
                loss_num = loss_num + loss_o
                loss_num_l2 = loss_num_l2 + L2_loss
            loss_av = loss_num + loss_av
            loss_num_l2_av = loss_num_l2_av + loss_num_l2
            optimizer.step()
            if i%8 == 7:
                print('[epoch: %d batch: %.2f]' % (epoch, i/ batch_num*100), ' [loss_av: %.4f] [loss_2: %.5f]' % (loss_av.data[0]/8, loss_num_l2_av.data[0]/8))
                loss_av = 0
                loss_num_l2_av = 0
#            points, target = data
#            points, target = Variable(points), Variable(target)
#            points = points.transpose(2, 1)
#            points, target = points.cuda(), target.cuda()
#            optimizer.zero_grad()
#            pred, trans = net(points)
#
#            pred = pred.view(-1, 2)
#            target = target.view(-1)
#
#            weight = np.array([1.0, 1.0])
#            weight = torch.from_numpy(weight)
#            weight = weight.float().cuda()
#            criterion = nn.CrossEntropyLoss(weight=weight)
#
#            loss = criterion(pred, target) * 10
#            norm = torch.matmul(trans, trans.transpose(2, 1))
#            I = Variable(torch.eye(64).repeat(loss.size()[0], 1, 1)).cuda()
#            L2_loss = torch.norm(norm - I, 2)                                  # L2 loss: norm(I - A*(AT)); A: feature alignment matrix
#            loss += L2_loss
#
#            loss.backward()
#            optimizer.step()
#
#            pred_choice = pred.data.max(1)[1]                                  # set the most probable one to 1

#            correct += pred_choice.eq(target.data).cpu().sum()                 # sum the right prediction

#            # TP (true positive): predict = 1, label = 1
#            TP += ((pred_choice == 1) & (target.data == 1)).cpu().sum()
#            # TN (true negative): predict = 0, label = 0
#            TN += ((pred_choice == 0) & (target.data == 0)).cpu().sum()
#            # FN (false positive): predict = 0, label = 1
#            FN += ((pred_choice == 0) & (target.data == 1)).cpu().sum()
#            # FP (false negative): predict = 1, label = 0
#            FP += ((pred_choice == 1) & (target.data == 0)).cpu().sum()
#
#            p = TP / (TP + FP)
#            r = TP / (TP + FN)
#            F1 = 2 * r * p / (r + p)
#            acc = (TP + TN) / (TP + TN + FP + FN)
#            print('batch_index: [%d/%d]' % (batch_index, len(dataloader)),
#                  'Train epoch: [%d]' % (epoch),
#                  'Loss %.4f L2_loss %.4f' % (loss.data[0], L2_loss.data[0]),
#                  'acc:%.4f p:%.4f r:%.4f F1:%.4f' % (acc, p, r, F1))
#            print('batch_index: [%d/%d]' % (batch_index, len(dataloader)),
#                  'Train epoch: [%d]' % (epoch),
#                  'Loss %.4f L2_loss %.4f' % (loss.data[0], L2_loss.data[0]))

        checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        utils.save_checkpoint(checkpoint, 'models/zan/myModels' + str(epoch) + '.pth')
        print('model saved')

