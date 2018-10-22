"""
loadOneVelodyneData   读取一副雷达数据 返回点云numpy
loadOneVelodyneLabel  读取label文件
loadCalib             读取calib 文件
process               对数据进行处理， 只保留正前方一个扇形区域内的点
transData             将KITTI数据读取，剪裁，转存
loadData              载入转存后的数据，用于训练
KITTI_Dateset pytorch 数据集对象

雷达坐标：x：正前方；y：左侧；z：正上方
"""

import os
import copy
import struct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

from scipy.stats import mode
from sklearn.cluster import MeanShift

'''定义可视区域矩形的大小及起始位置'''
window_L = 40
window_W = 30
window_H = 0.5
start_x  = 3

# Grid-Var
var_min = 0.05
point_number_of_grid = 50

Category = {'Pedestrian': 1,
               'Car': 2,
               'Cyclist': 3,
               'Truck': 4,
               'Misc': 5,
               'Van': 6,
               'Tram': 7,
               'Person_sitting': 8}

'''------------------------------- function --------------------------------'''

def loadOneVelodyneData(path='/home/xh/guazai/dataset/KITTI/training/velodyne/', tracking='0000',order=0):
    """
    load data
    """
    
    order = str(order)
    order = '0'*(6 - len(order)) + order
#    file_adress = path + order + '.bin'
    file_adress = path + tracking + '/' + order + '.bin'

    with open(file_adress, 'rb') as file:
        datas = file.read()
        length = len(datas)
        datas = struct.unpack('f' * int(length/4), datas)                      # binary -> float32
        datas = np.array(datas)
        datas.shape = -1, 4
        datas = datas[:, :3]
        return datas                                                           # dim: (len(datas), 3 (x, y, z))

def loadOneVelodyneLabel(path='/home/xh/guazai/dataset/KITTI/training/label_02/', tracking='0000',order=0):
    """
    load labels
    3D_height, 3D_width, 3D_length: object dimension
    x, y, z: object location
    """
    
    file_adress = path + tracking + '.txt'
    labels = pd.read_table(file_adress, header=None, delim_whitespace=True)
    
    # columns name
    labels.columns = ['frame',
                      'track_id',
                      'type',
                      'truncated',
                      'occluded',
                      'alpha',
                      'bbox_left',
                      'bbox_top',
                      'bbox_right',
                      'bbox_bottom',
                      '3D_height',
                      '3D_width',
                      '3D_length',
                      'x',
                      'y',
                      'z',
                      'rotation_y']
    
    # leave out DontCare
    labels = labels[labels['type'] != 'DontCare']

    # drop out data
    labels = labels.drop(['truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom'], axis=1)

    # transform to numpy
    labels = labels.values
    
    obj = []
    for label in labels:
        # label: [frame, track_id, type, 3D_height, 3D_width, 3D_length, x, y, z, rotation_y]
        ry = label[-1]
        R = [[np.cos(ry) , 0, np.sin(ry)],
             [      0 , 1,       0],
             [-np.sin(ry), 0, np.cos(ry)]]
        R = np.array(R)

        # 3D height, width, length
        l = label[-5]
        w = label[-6]
        h = label[-7]

        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners_3D = [x_corners, y_corners, z_corners]                         # coordinates of each corner of gt bounding box
        corners_3D = np.array(corners_3D)
        
        corners_3D = np.matmul(R, corners_3D)                                  # matrix multiply

        corners_3D[0, :] += label[-4]                                          # label[-4]: x
        corners_3D[1, :] += label[-3]                                          # label[-3]: y
        corners_3D[2, :] += label[-2]                                          # label[-2]: z

        # 转换到雷达坐标下
        velodyne_3D = np.zeros(corners_3D.shape)                               # 3 x 8
        velodyne_3D[0, :] = corners_3D[2, :] + 0.27     # x = z + 0.27
        velodyne_3D[1, :] = -corners_3D[0, :]           # y = -x
        velodyne_3D[2, :] = -corners_3D[1, :]           # z = -y

        obj.append([label[0], label[1], label[2], velodyne_3D, ry])            
    
    return obj                                                                 # dim: (len(labels), 5 (frame, track_id, type, velodyne_3D (3 x 8), rotation_y))

def loadCalib(path='/home/xh/guazai/dataset/KITTI/training/calib/', order=0):
    '''
    load calibration data
    '''
    order = str(order)
    order = '0' * (6 - len(order)) + order
    file_adress = path + order + '.txt'
    calib = pd.read_table(file_adress, header=None, delim_whitespace=True)
    print(calib)

def calc_y(x):
    '''
    用于计算扇形
    '''
    angle = 50
    angle = np.pi * angle /180
    return x * np.tan(angle)

def slide_window(data,N=[40,30]):
    '''
    input data: 3 < x < 43, -15 < y < 15, 0 < z <0.5
    '''
    gridpoint_x = np.linspace(start_x, start_x + window_L, N[0]+1)             # gridpoint_x = [3, 4, ...,  43]
    gridpoint_y = np.linspace(-window_W/2, +window_W/2, N[1]+1)                # gridpoint_y = [-15, -14, ..., 15]
    var_plan = np.ones(N) * (-10)
    data_out = np.zeros([1,3])
    grid_index = []
    for x in range(N[0]):
        for y in range(N[1]):
            index = (data[:, 0] > gridpoint_x[x]) & \
                    (data[:, 0] < gridpoint_x[x + 1]) & \
                    (data[:, 1] > gridpoint_y[y]) & \
                    (data[:, 1] < gridpoint_y[y + 1])                          #index dim: (len(data), 1)

            temp = data[index]                                                 # data that in (x, y) grid
            if len(temp) > point_number_of_grid:                               # point_number_of_grid = 50
                var_plan[x, y] = np.var(temp[:, 2])                            
                if var_plan[x, y] > var_min:                                   # var_min = 0.05, 不是地面
                    maxData = np.max(temp[:, 2])                               # maxData: max z
                    minData = np.min(temp[:, 2])                               # minData: min z

                    temp = temp[temp[:, 2] > minData + 0.05 * (maxData - minData)]  # required z: z > [minz + 0.05 * (maxz - minz)]
                    if len(temp) > point_number_of_grid:
                        data_out = np.append(data_out, temp, axis=0)
                        grid_index.append(len(temp))                           # number of data conforming to requirements in each grid

    return data_out[1:], grid_index

def myMeanShift(data, bandwidth=0.5):
    '''
    mean shift clustering
    '''
    x = data[:, :2]
    ms = MeanShift(bandwidth=0.5, bin_seeding=True)
    ms.fit(x)                                                                  # perform cluster
    labels = ms.labels_                                                        # label of each point
    cluster_centers = ms.cluster_centers_                                      # cluster center
    return labels, cluster_centers

def process(data, labels):
    """
    preporcess raw data
    
    用Grid-Var去除地面，用MeanShift进行聚类，丢弃不在检测范围内的数据，用label进行数据分割
    
    input:
    data: [x, y, z]
    labels: [frame, track_id, type, velodyne_3D, rotation_y]
    
    return: 
    data, color     data.shape=[-1,3] category.shape=[-1,3]
    """
    
    ## select points in appointed regions    
    # select x: 3 < x < 43
    data = data[data[:, 0] > start_x]                                         
    data = data[data[:, 0] < 43]                                              
    # select y: |y| < 15
    data = data[np.abs(data[:, 1]) < window_W/2]                               
    # select z: z < 0.5
    data = data[data[:, 2] < window_H]
    # 只取一个张角为2*angle的扇形
    data = data[np.abs(data[:, 1]) < calc_y(data[:, 0])]
    # grid-var 去除地面
    data, grid_index = slide_window(data)

    N = len(data)
    category = np.zeros([N, 1], dtype=np.int)
    # copy_data = copy.deepcopy(data[:, :3])
    # I = np.array(range(0, N)).reshape([-1, 1])
    # copy_data = np.append(copy_data, I, axis=1)
    # grid_labels = np.zeros([len(grid_index), 1], dtype=np.int)

    # label all cloud points according to labels
    for label in labels:                                                       # each label: [frame, track_id, type, velodyne_3D (3 x 8), rotation_y]
        if label[2] in ['Pedestrian', 'Car', 'Cyclist', 'Truck', 'Misc', 'Van', 'Tram', 'Person_sitting']:
            # 先用较为粗暴的方法试一试
            maxData = np.max(label[3], axis=1)                                 # max (x, y, z) in velodyne_3D
            minData = np.min(label[3], axis=1)                                 # min (x, y, z) in velodyne_3D

            index = (maxData > data[:, :3]) & (data[:, :3] > minData)          
            index = index[:, 0] & index[:, 1] & index[:, 2]
            category[index] = Category[label[2]]                               # label the True cloud points with categary (1, ..., 8), label 0 to other points

#            index_cut = index[:grid_index[0]]
#            if np.sum(index_cut) > point_number_of_grid/2:
#                grid_labels[0] = Category[label[2]]
#            for k in range(1, len(grid_labels)):
#                index_cut = index[sum(grid_index[:k - 1]):sum(grid_index[:k])]
#                if np.sum(index_cut) > point_number_of_grid/2:
#                    grid_labels[k] = Category[label[2]]

    # clustering
    cluster_labels, cluster_centers = myMeanShift(data, bandwidth=0.3)         # len(cluster_labels) == len(data)
    '''
    cluster_labels: denote which cluter each point belongs to
    cluster_centers: center coordinate of every cluster
    '''
    
    # 在利用聚类后的结果进行一次去除地面
    cluster_number = np.max(cluster_labels) + 1                                # number of clusters
    keep_number = []                                                           # store the kept cluster

    for k in range(cluster_number):
        if np.sum(cluster_labels == k) > 100:                                  # 点数过少的簇不要
            if np.var(data[cluster_labels == k, 2]) > 0.01:                    # 若第k个簇的z方向方差过小，说明是地面，丢弃
                keep_number.append(k)

    for k in range(cluster_number):                                            # 将不要的label改为-1
        if k not in keep_number:
            cluster_labels[cluster_labels == k] = -1                           
    index = cluster_labels >= 0
    data = data[index]                                                         # leave out the unused data
    category = category[index]
    cluster_labels = cluster_labels[index]
    cluster_centers = cluster_centers[keep_number]
    return data, category, cluster_labels, cluster_centers                     # category (0, ..., 8), cluster_labels: category and cluster label of each cloud point (0, ..., 8)

def classifyRawData(data, labels):
    '''
    classfy each point in data acoording to labels
    '''
    
    category = np.zeros([len(data), 1])
    
    for label in labels:                                                       # each label: [frame, track_id, type, velodyne_3D (3 x 8), rotation_y]
        if label[2] in ['Pedestrian', 'Car', 'Cyclist', 'Truck', 'Misc', 'Van', 'Tram', 'Person_sitting']:
            maxData = np.max(label[3], axis=1)                                 # max (x, y, z) in velodyne_3D
            minData = np.min(label[3], axis=1)                                 # min (x, y, z) in velodyne_3D

            index = (maxData > data[:, :3]) & (data[:, :3] > minData)          
            index = index[:, 0] & index[:, 1] & index[:, 2]
            category[index] = Category[label[2]]                               # label the True cloud points with categary (1, ..., 8), label 0 to other points
    return category

def transData(root='/home/xh/guazai/dataset/KITTI/training/', data_dir='velodyne/', tracking='0000', label_dir='label_2/', aim_path='/home/xh/guazai/dataset/KITTI/preprocessedData/'):
    '''
    transform raw data in original files to processed data, and then save as .npy file
    '''
    
    for roots, dirs, names in os.walk(root + data_dir + tracking + '/'):       # traverse raw data in each tracking
        files = names
    number = len(files)                                                                                                             
    data_struct = []
    label_struct = []
    for N in range(number):                                                    # N-th frame in each tracking
        
        n = str(N)
        n = '0' * (6-len(n)) + n
        if not os.path.isfile('/home/xh/guazai/dataset/KITTI/training/velodyne/' + tracking + '/' + n + '.bin'):
            continue
        
        data = loadOneVelodyneData(tracking=tracking, order=N)
        labels = loadOneVelodyneLabel(order=N, tracking=tracking)
        index = [i for i in range(len(labels)) if labels[i][0] == N]           
        labels_frame = []
        for i in index:
            labels_frame.append(labels[i])                                     
        
        # data, labels_frame: N-th frame data and labels
        data, category, cluster_labels, cluster_centers = process(data, labels_frame)
        
        # appoint clustered data to the category which most points belong to
        if len(cluster_labels) > 0:
            for k in range(np.max(cluster_labels) + 1):
                index = (cluster_labels == k)
                if np.sum(index) != 0:
                    data_struct.append(data[index])
                    most_frequent = mode(category[index])[0][0][0]
                    label_struct.append(most_frequent)
            print('tracking: %s: data %d/%d has loaded. counts: %d' % (tracking, N, number, len(label_struct)))

    # length: number x cluster_in_each_N
    data_struct = np.array(data_struct)                                        
    label_struct = np.array(label_struct)
    
    print('data size', data_struct.shape, 'label size', label_struct.shape)
    np.save(aim_path + tracking + '/' + 'data.npy', data_struct)
    np.save(aim_path + tracking + '/' + 'label.npy', label_struct)

def loadData(root='/home/xh/guazai/dataset/KITTI/preprocessedData/', tracking='0000'):
    '''
    load data of appointed tracking
    data have been transformed from original files
    '''
    
    path = root + tracking + '/'
    datas = np.load(path + 'data.npy')
    labels = np.load(path + 'label.npy')
    return datas, labels

def load_all_data(root='/home/xh/guazai/dataset/KITTI/preprocessedData/'):
    '''
    load all data
    data have been transformed from original files
    '''
    
    path = root
    datas = np.load(path + 'data.npy')
    labels = np.load(path + 'label.npy')
    return datas, labels

def criterion_points(data):
    '''
    resize number of data to n
    '''
    
    length = len(data)
    temp = data
    if length < n:
        times = np.floor(n/length)
        times = np.int(times)
        while times:
            data = np.append(data, temp, axis=0)
            times -= 1
        data = data[:n]
    if length > n:
        random_index = np.random.randint(0, length, n)
        data = data[random_index]
    return data

def combination(aim_path='/home/xh/guazai/pointnet/pointnetcode/preprocessedData/'):
    '''
    combine data of all trackings
    '''
    
    trackings=['0000', '0002', '0003', '0004', '0006', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
    
    datas, labels = loadData(tracking=trackings[0])
    print(datas.shape, labels.shape)
    for k in range(1, len(trackings)):
        data, label = loadData(tracking=trackings[k])
        datas = np.append(datas, data, axis=0)
        labels = np.append(labels, label, axis=0)
        print(datas.shape, labels.shape)

    np.save(aim_path + 'data.npy', datas)
    np.save(aim_path + 'label.npy', labels)

    print('combine complated, totally %d objects' % (datas.shape[0]))
    
def save_checkpoint(state, filename='models/pointnetV2.0.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename='models/pointnetV2.0.pth.tar'):
    return torch.load(filename)

def save_params(filename):
    return torch.load(filename)

def load_params(state_dict, filename):
    torch.save(state_dict, filename)

'''------------------------------ class ------------------------------------'''
  
class KITTI_Dateset(data.Dataset):
    def __init__(self, tracking='0000', all_data=False):
        if all_data:
            datas, labels = load_all_data()
        else:
            datas, labels = loadData(tracking=tracking)
        labels.shape = -1, 1
#        datas_norm = []
#        for data in datas:
#            data = criterion_points(data, n=500)
#            datas_norm.append(data)
#        datas_norm = np.array(datas_norm)
#        datas_norm = np.float32(datas_norm)

        for k in range(len(labels)):
            if labels[k][0] != 1:
                labels[k][0] = 0

        self.datas = datas
        self.labels = labels

        # self.datas = torch.from_numpy(datas)
        # self.labels = torch.from_numpy(labels)


    def __getitem__(self, index):
        return torch.from_numpy(self.datas[index]).float(), torch.from_numpy(self.labels[index]).long()

    def __len__(self):
        return len(self.datas)

if __name__ == '__main__':
    combination()
    