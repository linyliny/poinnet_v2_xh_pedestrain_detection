"""
PointNet
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

'''first T-net; input: cluster x num_points x 3'''
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512, 256)
        self.fc3   = nn.Linear(256, 9)
        self.relu  = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
#        self.bn4 = nn.BatchNorm1d(1)
#        self.bn5 = nn.BatchNorm1d(1)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))                                    
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        input_size = x.size()[2]
        x = F.max_pool1d(x, input_size)                                        # output dim: (cluster/batch, 1024, 1)
        x = x.view(-1, 1024)                                                  
#        x = F.relu(self.bn4(self.fc1(x)))
#        x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                                                        # dim: (cluster/batch, 9)
        iden = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32).reshape([1, 9]).repeat(batchsize, axis=0)
        iden = Variable(torch.from_numpy(iden))
        if x.is_cuda:
            iden = iden.cuda()
        x += iden
        x = x.view(-1, 3, 3)                                                   # dim: (cluster/batch, 3, 3)
        return x

'''second T-net; input: cluster x num_points x 64'''
class STN64d(nn.Module):
    def __init__(self):
        super(STN64d, self).__init__()
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64*64)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
#        self.bn4 = nn.BatchNorm1d(512)
#        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        input_size = x.size()[2]
        x = F.max_pool1d(x, input_size)
#        x = self.mp1(x)
        x = x.view(-1, 1024)                                                   # dim of output: (cluster/batch, 1024) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                                                        # dim of output: (cluster/batch, 64*64)
        iden = np.eye(64).astype(np.float32).reshape([1, 64*64]).repeat(batchsize, axis=0)
        iden = Variable(torch.from_numpy(iden))

        if x.is_cuda:
            iden = iden.cuda()
        x += iden
        x = x.view(-1, 64, 64)                                                 # dim: (cluster/batch, 64, 64)
        return x

'''classification net'''
class PointNetfeat(nn.Module):
    def __init__(self):              
        super(PointNetfeat, self).__init__()
        
        self.stn3  = STN3d()
        self.stn64 = STN64d()
#        self.mp1   = torch.nn.MaxPool1d()
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn128 = nn.BatchNorm1d(128)
        self.bn1024 = nn.BatchNorm1d(1024)


    def forward(self, x):
#        batchsize = x.size()[0]
#        input_size = x.size()[1]
        trans = self.stn3(x)                                                   # dim: x: (cluster/batch, 3, num_points); trans: (cluster/batch, 3, 3)                                           
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)                                                # batch matrix-matrix product
        x = x.transpose(2, 1)                                                  # dim: (cluster/batch, 3, num_points)

        x = F.relu(self.bn64(self.conv1(x)))                                   
        x = F.relu(self.bn64(self.conv2(x)))                                   

        trans = self.stn64(x)                                                  # dim: x: (cluster/batch, 64, num_points); trans:(cluster/batch, 64, 64)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)                                                  # dim: (cluster/batch, 64, num_points)

        x = F.relu(self.bn64(self.conv2(x)))
        x = F.relu(self.bn128(self.conv3(x)))
        x = self.bn1024(self.conv4(x))
        input_size = x.size()[2]
        x = F.max_pool1d(x, input_size)
#        x = self.mp1(x)

        return x, trans                                                        # dim: x: (cluster/batch, 1024, 1); trans: (cluster/batch, 64, 64)

'''PointNet classification network'''
class PointNet(nn.Module):
    def __init__(self, k=2):                                   # num_points: number of input points
        super(PointNet, self).__init__()
        self.k = k
        self.feat = PointNetfeat()
        
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, self.k)

        self.bn0 = nn.BatchNorm1d(3)
#        self.bn1 = nn.BatchNorm1d(512)
#        self.bn2 = nn.BatchNorm1d(256)
#        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
#        batchsize = x.size()[0]
        x = self.bn0(x)                                                        # input dim: (cluster/batch, 3, num_points)
        x, trans = self.feat(x)

        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)                                                      # point feature
        x = x.view(-1, self.k)                                                 # dim: (cluster/batch, k); k: number of class

        x = F.log_softmax(x, dim=1)
        return x, trans
