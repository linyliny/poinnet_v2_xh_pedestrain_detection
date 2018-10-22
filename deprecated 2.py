"""
KITTI数据集转换文件
transDate 将KITTI数据读取，剪裁，转存
loadData    载入转存后的数据，用于训练
KITTI_Dateset pytorch数据集对象
"""
import torch
import torch.utils.data as data

import kitti
import os
import numpy as np
from scipy.stats import mode

def transDate(root='E:\\KITTI\\tracking\\training\\', data_dir='velodyne\\', tracking='0016', label_dir='label_2\\', aim_path='KITTI_data\\'):
    for roots, dirs, names in os.walk(root + data_dir + tracking + '\\'):
        files = names
    number = len(files)
    data_struct = []
    label_struct = []
    for N in range(number):
        data = kitti.loadOneVelodyneData(tracking=tracking, order=N)
        labels = kitti.loadOneVelodyneLabel(order=N, tracking=tracking)
        index = [i for i in range(len(labels)) if labels[i][0] == N]
        labels_frame = []
        for i in index:
            labels_frame.append(labels[i])
        data, category, cluster_labels, cluster_centers = kitti.process(data, labels_frame)
        if len(cluster_labels) > 0:
            for k in range(np.max(cluster_labels) + 1):
                index = cluster_labels == k
                if np.sum(index) != 0:
                    data_struct.append(data[index])
                    most_frequent = mode(category[index])[0][0][0]
                    label_struct.append(most_frequent)
            print('data %d/%d has loaded. counts: %d' % (N, number, len(label_struct)))

    data_struct = np.array(data_struct)
    label_struct = np.array(label_struct)
    print('data size', data_struct.shape, 'label size', label_struct.shape)
    np.save(aim_path + tracking + '\\' + 'data.npy', data_struct)
    np.save(aim_path + tracking + '\\' + 'label.npy', label_struct)

def loadData(root='KITTI_data\\', tracking='0016'):
    path = root + tracking + '\\'
    datas = np.load(path + 'data.npy')
    labels = np.load(path + 'label.npy')
    return datas, labels

def criterion_points(data, n=1000):
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

class KITTI_Dateset(data.Dataset):
    def __init__(self, tracking='0019', all_data=False):
        if all_data:
            datas, labels = load_all_data()
        else:
            datas, labels = loadData(tracking=tracking)
        labels.shape = -1, 1
        datas_norm = []
        for data in datas:
            data = criterion_points(data, n=500)
            datas_norm.append(data)
        datas_norm = np.array(datas_norm)
        datas_norm = np.float32(datas_norm)

        for k in range(len(labels)):
            if labels[k][0] != 1:
                labels[k][0] = 0

        self.datas = datas_norm
        self.labels = labels

        # self.datas = torch.from_numpy(datas)
        # self.labels = torch.from_numpy(labels)


    def __getitem__(self, index):
        return torch.from_numpy(self.datas[index]), \
               torch.from_numpy(self.labels[index]).long()

    def __len__(self):
        return len(self.datas)

def combination(trackings=['0013', '0015', '0019'], aim_path='KITTI_data\\'):
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

def load_all_data(root='KITTI_data\\'):
    path = root
    datas = np.load(path + 'data.npy')
    labels = np.load(path + 'label.npy')
    return datas, labels

if __name__ == '__main__':
    # combination()
    tracking = '0013'
    # transDate(tracking=tracking)
    # datas, labels = loadData(tracking=tracking)
    train = KITTI_Dateset(tracking=tracking, all_data=True)
    print(len(train))
