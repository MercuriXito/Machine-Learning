# -*- coding:utf-8 -*-
"""
@author victor chen
@time 2018/6/12
"""

import numpy as np

class KMeans(object):
    '''
    Args: 
        distance: (defalut or wrong paramerter: using 'euclid')
            'euclid','manhattan','chebyshev'
    '''
    def __init__(self,n_clusters,distance = 'euclid', precise = 1e-5, max_iternum = 500,**param):
        self.n_clusters = n_clusters
        self.distance = distance
        self.precise = precise
        self.max_iternum = max_iternum
        self.distance_fun = {
            'euclid': self.Euclid_distance,
            'manhattan':self.Manhattan_distance,
            'chebyshev':self.Chebyshev_distance,
        }
        
    def fit(self, X):
        # 参数初始化
        self.X = X
        precise = self.precise
        max_iternum = self.max_iternum
        self.m,self.n = X.shape
        self.k = self.n_clusters
        # cal_distance 变为计算相应距离的函数
        try:
            self.cal_distance = self.distance_fun[self.distance]
        except KeyError as e:
            print 'Distance is not supported, change to default \'Euclidean Distance\''
            self.cal_distance = self.Euclid_distance

        # 随机初始中心,(此处的处理比较简单)
        # print X.min(),X.max()
#         low , high = X.min(), X.max()
#         initial_centers = np.random.random(size = (self.k,self.n))
#         initial_centers = initial_centers * (high - low) + low
        initial_centers = self.initial_c()
        # print initial_centers
        self.centers = initial_centers
        # 第一次迭代
        # 计算点与中心的距离
        self.cal_distance()
        # 点分类
        self.tag_sort()
        # 簇中心移动
        self.centroids()
        present_var = 0 
        loopi = 0
        # 循环迭代
        while( np.abs(present_var - self.var) > precise and loopi <= max_iternum):
            present_var = self.var
            # 更新距离矩阵
            self.cal_distance()
            # 更新点的簇，并更新var
            self.tag_sort()
            # 更新簇中心
            self.centroids()
            
            loopi += 1
        self.iternum = loopi + 1
        
    # KMeans++ 选取聚类中心
    def initial_c(self):
#         choose = np.random.randint(low = 0, high = self.m)
        # 固定选择第一个数据作为第一个聚类中心点
        choose = 0
        center = np.zeros( (self.k,self.n) )
        # 随机选取第一个聚类中心
        center[0,:] = self.X[choose,:]
        # 选取剩余的聚类中心
        for i in range(1, self.k):
            temparr = np.zeros((self.m, i))
            # 计算每个点到当前已有的聚类中心的最小值
            for j in range(i):
                temparr[:,j] = np.linalg.norm(self.X - center[j,:], axis = 1)
            temparr = np.min(temparr, axis = 1)
#             t = np.random.random() * temparr.sum()
            t = temparr.sum() * 0.8
            tempsort = temparr.argsort()
            index = 0
            while(t > 0):
                t = t - temparr[tempsort[self.m - index - 1]]
                index += 1
            center[i,:] = self.X[tempsort[self.m - index - 1],:]
        return center
        
    # 簇分配，计算SSE
    def tag_sort(self):
        _,ind = np.unravel_index(np.argmin(self.distance, axis=1), self.distance.shape)
        self.labels_ = ind
        print ind
        self.var = np.min(self.distance,axis = 1).sum()

    # 重新计算簇中心
    def centroids(self):
        for i in range(self.n_clusters):
            # print sum(self.labels_ == i)
            self.centers[i,:] = np.mean(self.X[self.labels_ == i], axis = 0)
        # 空簇检测
        null_label = []
        for i in range(self.n_clusters):
            if self.centers[i,0] == 'nan':
                null_label.append(i)
        if null_label == []:
            return
        
                
        # print self.centers
            
    # 计算欧式距离,需要更新self.distance: shape()
    def Euclid_distance(self):
        m,n = self.m,self.n
        k = self.k
        distance = np.zeros((m,k))
        for i,center in enumerate(self.centers):
            each_distance = np.linalg.norm(self.X - center, axis = 1)
            distance[:,i] = each_distance
        self.distance = distance
    
    # 计算曼哈顿距离
    def Manhattan_distance(self):
        m,n = self.m,self.n
        k = self.k
        distance = np.zeros((m,k))
        for i,center in enumerate(self.centers):
            each_distance = np.sum(np.abs(self.X - center), axis = 1)
            distance[:,i] = each_distance
        self.distance = distance

    # 计算切比雪夫距离
    def Chebyshev_distance(self):
        m,n = self.m,self.n
        k = self.k
        distance = np.zeros((m,k))
        for i,center in enumerate(self.centers):
            each_distance = np.max(np.abs(self.X - center), axis = 1)
            distance[:,i] = each_distance
        self.distance = distance
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    X = np.random.random(size = (1200,2))
    param = {
        'n_clusters' : 5,
        'distance' : 'euclid',
    }
    k = KMeans( **param)
    k.fit(X)
    plt.title('example data')
    plt.scatter(X[:,0],X[:,1],c = k.labels_)
    plt.scatter(k.centers[:,0],k.centers[:,1],marker = '*')
        