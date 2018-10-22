import kitti
import numpy as np
import PointCloudShow
import matplotlib.pyplot as plt
import sys
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块

tracking = '0019'
order = 234

#                  0          1         2           3
color_cate = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
              [1, 1, 0], [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0],
              [0.5, 0, 0.5], [0, 0.5, 0.5]]
if __name__ == '__main__':
    # read data
    data = kitti.loadOneVelodyneData(tracking=tracking, order=order)     # [x, y, z, intensity]
    labels = kitti.loadOneVelodyneLabel(order=order, tracking=tracking)  # [frame, track_id, type, velodyne_3D, rotation_y]
    index = [i for i in range(len(labels)) if labels[i][0] == order]
    labels_frame = []
    for i in index:
        labels_frame.append(labels[i])
        
    '''preprocess data'''
    data, category, cluster_labels, cluster_centers = kitti.process(data, labels_frame)

    N = len(data)
    color = np.zeros([N, 3])
    for n in range(N):
        color[n] = color_cate[category[n, 0]]

    # plt.figure()
    # ret = plt.hist(data[:, 2], 50)
    # plt.show()

    data[:, 0] -= 10


    print(np.max(cluster_labels))
    index = cluster_labels == 3
    data = data[index]
    color = color[index]

    print(len(cluster_centers), 'objects ')
    print('point number:', len(data))
    PointCloudShow.Image(data, color)

