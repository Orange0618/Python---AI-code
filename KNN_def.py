import numpy as np
#归一化，避免某些特征数值太大，主导距离计算
def normalize(X,mean,std):
    return (X-mean)/std
#训练数据
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
#归一化训练数据
xTrain = normalize(xTrain, mean, std)
xTest = normalize(xTest,mean,std)


#计算欧式距离
def euclidean_distance(x,y):
    diff = x-y
    diff_2 = diff**2
    summ = np.sum(diff_2)
    dist = summ**0.5
    return dist

#K-近邻分类器
def knn_classifier(input, trains, classes, k=3):
    dists = [ ]
    for i in range(len(trains)):
        dists.append([i,euclidean_distance(input,trains[i])])
    dists.sort(key=lambda x:x[1])   #对列表按距离排序
    class_counts = { }   #以类别为key，以类别数量为value
    for i in range(k):
        neighbor_index = dists[i][0]
        class_ = classes[neighbor_index]
        if class_ not in class_counts:
            class_counts[class_] = 0
        class_counts[class_] +=1
    max_label_count = 0
    for class_ in class_counts:
        if max_label_count < class_counts[class_]:
            max_label_count = class_counts[class_]
            predicted_class = class_
    return predicted_class

print("预测类别数：",end=" ")
for test1 in xTest:
    predicted_class = knn_classifier(test1, xTrain, yTrian, k = 6)
    print(predicted_class,end=" ")
print("实际类别是：",yTest)