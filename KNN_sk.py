import numpy as np
from sklearn.neighbors import NearestNeighbors
def normalize(X, mean, std):
    return (X-mean)/std
xTrain = np.array([[3.32,94],[3.05,120],[3.70,160],
                   [3.52,170],[3.60,155],[3.36,78],
                   [2.70,75],[2.90,80],[3.12,100],
                   [2.80,125]])
yTrian = np.array([0,0,0,0,0,1,1,1,1,1])
xTest = np.array([[3.00,100],[3.25,93],[3.63,163],
                  [2.82,120],[3.37,89]])
yTest = np.array([1,0,1,1,1])
mean = xTrain.mean(axis=0)
std = xTrain.std(axis=0)
xTrain = normalize(xTrain,mean,std)
xTest = normalize(xTest,mean,std)

#创建模型，使用暴力法实现K-近邻分类算法
model = NearestNeighbors(n_neighbors=3,algorithm="brute", metric="euclidean")
model.fit(xTrain)
distance, indices = model.kneighbors(xTest)
print("最近邻的索引是:",indices)
print("对应的最近K个类别是:",yTrian[indices])
