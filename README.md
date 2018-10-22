# poinnet_v2_xh_pedestrain_detection
cloud points预处理：
1. 筛选点(x, y, z)符合 3 < x < 43, -15 < y < 15, 0 < z <0.5，
2. 用slide_window筛选；slide_window：将摄像机前方划分为40x30的网格，要求每一个网格中data中的z符合：z > [minz + 0.05 * (maxz - minz)]，且z不属于地面
3. 根据labels标记每个点的category
4. 通过meanshift进行聚类处理
5. 聚类后的点，点数过少的cluster丢弃，z方差过小的cluster（地面）丢弃
完成预处理后，每个点具有category和cluster的属性
在形成.npy文件和在show.py中，对于每个cluster中的所有点，将他们的category全改为这些点中属于某个category最多的category；

----------------------------------------------------------------------------------

点云坐标：(x, y, z)：车辆正前方，左侧，正上方
training 0001数据不完整
训练时使用的是处理过的数据进行训练
iterations = total_samples / batch_size 

----------------------------------------------------------------------------------

缺少dropout
