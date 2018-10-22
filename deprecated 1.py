"""
kitti
数据集读取文件
loadOneVelodyneData 读取一副雷达数据 返回点云numpy
loadOneVelodyneLabel 读取label文件
loadCalib           读取calib 文件
process             对数据进行处理， 只保留正前方一个扇形区域内的点
"""
import numpy as np
import struct
import pandas as pd
import copy
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift

cos = np.cos
sin = np.sin

def loadOneVelodyneData(path='/media/wwch/data/data/KITTI/training/velodyne/', tracking='0017',order=0):
    """
    load data
    return:  shape = [-1, 4], [x, y, z, r]
    """
    order = str(order)
    order = '0'*(6 - len(order)) + order
    file_adress = path + tracking + '/' + order + '.bin'

    with open(file_adress, 'rb') as file:
        datas = file.read()
        length = len(datas)
        datas = struct.unpack('f' * int(length/4), datas)                      # binary -> float32
        datas = np.array(datas)
        datas.shape = -1, 4
        return datas

def loadOneVelodyneLabel(path='/media/wwch/data/data/KITTI/training/label_02/', tracking='0016',order=73):
    """
    load labels
    """
    file_adress = path + tracking + '.txt'
    labels = pd.read_table(file_adress, header=None, delim_whitespace=True)
    
    '''columns name'''
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
    
    '''leave out DontCare'''
    labels = labels[labels['type'] != 'DontCare']

    '''drop out data'''
    labels = labels.drop(['truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom'], axis=1)

    '''transform to numpy'''
    labels = labels.values
    obj = []
    for label in labels:
        '''label: [frame, track_id, type, 3D_height, 3D_width, 3D_length, x, y, z, rotation_y]'''
        ry = label[-1]
        R = [[cos(ry) , 0, sin(ry)],
             [      0 , 1,       0],
             [-sin(ry), 0, cos(ry)]]
        R = np.array(R)

        '''height, width, length'''
        l = label[-5]
        w = label[-6]
        h = label[-7]

        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners_3D = [x_corners, y_corners, z_corners]                         # coordinates of each corner of gt bounding box
        corners_3D = np.array(corners_3D)
        
        corners_3D = np.matmul(R, corners_3D)                                  # matrix multiply

        corners_3D[0, :] += label[-4] # label[-4]: x
        corners_3D[1, :] += label[-3] # label[-3]: y
        corners_3D[2, :] += label[-2] # label[-2]: z

        '''转换到雷达坐标下'''
        velodyne_3D = np.zeros(corners_3D.shape)                               # 3 x 8
        velodyne_3D[0, :] = corners_3D[2, :] + 0.27     # x = z + 0.27
        velodyne_3D[1, :] = -corners_3D[0, :]           # y = -x
        velodyne_3D[2, :] = -corners_3D[1, :]           # z = -y

        obj.append([label[0], label[1], label[2], velodyne_3D, ry])            # each label: [frame, track_id, type, velodyne_3D (3 x 8), rotation_y]

    return obj

def loadCalib(path='/media/wwch/data/data/KITTI/training/calib/', order=0):
    order = str(order)
    order = '0' * (6 - len(order)) + order
    file_adress = path + order + '.txt'
    calib = pd.read_table(file_adress, header=None, delim_whitespace=True)
    print(calib)



Category = {'Pedestrian': 1,
            'Car': 2,
            'Cyclist': 3,
            'Truck': 4,
            'Misc': 5,
            'Van': 6,
            'Tram': 7,
            'Person_sitting': 8}
angle = 50
angle = np.pi * angle /180

'''用于计算扇形'''
def calc_y(x):
    return x * np.tan(angle)

'''定义可视区域矩形的大小及起始位置'''
window_L = 40
window_W = 30
window_H = 0.5
start_x = 3

# Grid-Var
var_min = 0.05
point_number_of_grid = 50

def slide_window(data,N=[40,30]):
    '''data: 3 < x <43, -15 < y < 15, 0 < z <0.5'''
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

            temp = data[index]
            if len(temp) > point_number_of_grid:                               # point_number_of_grid = 50
                var_plan[x, y] = np.var(temp[:, 2])                            
                if var_plan[x, y] > var_min:                                   # var_min = 0.05, 不是地面
                    maxData = np.max(temp[:, 2])                               # maxData: max z
                    minData = np.min(temp[:, 2])                               # minData: min z

                    temp = temp[temp[:, 2] > minData + 0.05 * (maxData - minData)]  # required z: z > [minz + 0.05 * (maxz - minz)]
                    if len(temp) > point_number_of_grid:
                        data_out = np.append(data_out, temp, axis=0)
                        grid_index.append(len(temp))                           # number of data conforming to requirements of each grid

    return data_out[1:], grid_index

'''
总体思路：
用Grid-Var去除地面
用MeanShift进行聚类
'''

def myMeanShift(data, bandwidth=0.5):
    '''
    mean shift clustering
    '''
    x = data[:, :2]
    ms = MeanShift(bandwidth=0.5, bin_seeding=True)
    ms.fit(x)                                              # perform cluster
    labels = ms.labels_                                    # label of each point
    cluster_centers = ms.cluster_centers_                  # cluster center
    return labels, cluster_centers

def process(data, labels):
    """
    丢弃不在检测范围内的数据，用label进行数据分割
    :param data:
    :param label:
    :return: data, color     data.shape=[-1,3] category.shape=[-1,3]
    
    data: [x, y, z]
    label: [frame, track_id, type, velodyne_3D, rotation_y]
    """

    data = data[:, :3]                                                         # (x, y, z)

    '''3 < x < 43'''
    data = data[data[:, 0] > start_x]                                         
    data = data[data[:, 0] < 43]                                              
    '''|y| < 15'''
    data = data[np.abs(data[:, 1]) < window_W/2]                               
    '''z < 0.5'''
    data = data[data[:, 2] < window_H]
    # 只取一个张角为2*angle的扇形
    data = data[np.abs(data[:, 1]) < calc_y(data[:, 0])]
    # grid-var 去除地面
    data, grid_index = slide_window(data)
    # 原始数据预处理完毕

    N = len(data)
    category = np.zeros([N, 1], dtype=np.int)
    # copy_data = copy.deepcopy(data[:, :3])
    I = np.array(range(0, N)).reshape([-1, 1])
    # copy_data = np.append(copy_data, I, axis=1)
    grid_labels = np.zeros([len(grid_index), 1], dtype=np.int)

    for label in labels:                                                       # each label: [frame, track_id, type, velodyne_3D (3 x 8), rotation_y]
        if label[2] in ['Pedestrian', 'Car', 'Cyclist', 'Truck', 'Misc', 'Van', 'Tram', 'Person_sitting']:
            # 先用较为粗暴的方法试一试
            maxData = np.max(label[3], axis=1)                                 # max (x, y, z) in velodyne_3D
            minData = np.min(label[3], axis=1)                                 # min (x, y, z) in velodyne_3D
            # set cloud points in gt box True
            index = (maxData > data[:, :3]) & (data[:, :3] > minData)          
            index = index[:, 0] & index[:, 1] & index[:, 2]
            category[index] = Category[label[2]]                               # label the True cloud points

            index_cut = index[:grid_index[0]]
            if np.sum(index_cut) > point_number_of_grid/2:
                grid_labels[0] = Category[label[2]]
            for k in range(1, len(grid_labels)):
                index_cut = index[sum(grid_index[:k - 1]):sum(grid_index[:k])]
                if np.sum(index_cut) > point_number_of_grid/2:
                    grid_labels[k] = Category[label[2]]

    cluster_labels, cluster_centers = myMeanShift(data, bandwidth=0.3)
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

    for k in range(cluster_number):                                            # 将不要的lable改为-1
        if k not in keep_number:
            cluster_labels[cluster_labels == k] = -1                           # problem?
    index = cluster_labels >= 0
    data = data[index]
    category = category[index]
    cluster_labels = cluster_labels[index]
    cluster_centers = cluster_centers[keep_number]
    return data, category, cluster_labels, cluster_centers
    '''
    data, category, cluster_labels, cluster_centers: selected
    '''

