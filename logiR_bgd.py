import numpy as np
from bgdLR import bgd_optimizer

#数据归一化
def normalize(X, mean, std):
    return (X-mean) / std

def make_ext(x):
    ones = np.ones(1)[:,np.newaxis]
    new_x = np.insert(x,0,ones,axis=1)  #第二个参数0表示在第0个位置插入
    return new_x

def logistic_fun(z):
    return 1./(1+np.exp(-z))

#定义损失函数，即要最小化的目标函数
def cost_fun(w,X,Y):     
    tmp = logistic_fun(X.dot(w))
    cost = -Y.dot(np.log(tmp)-(1-Y).dot(np.log(1-tmp)))
    return cost

#定义计算梯度函数
def grad_fun(w, X, Y):       
    grad = X.T.dot(logistic_fun(X.dot(w))-Y) / len(X)
    return grad

#训练数据
xTrain = np.array([[3.32, 94], [3.05, 120], [3.70, 160], 
                   [3.52,170], [3.60, 155], [3.36, 78], 
                   [2.70, 75], [2.90, 80], [3.12, 100],
                   [2.80, 125]])
yTrain = np.array([0,0,0,0,0,1,1,1,1,1])
mean = xTrain.mean(axis = 0)
std = xTrain.std(axis = 0)
xTrain_norm = normalize(xTrain, mean, std)
xTrain_ext = make_ext(xTrain_norm)

#随机初始化W
np.random.seed(0)
init_W = np.random.random(3)    #包括w0，w1，w2

#梯度下降求解
i,W = bgd_optimizer(cost_fun, grad_fun, init_W, xTrain_ext, yTrain, 
                    lr = 0.001, tolerance=1e-5, max_iter=1000000)
w0,w1,w2 = W

#输出结果
print(f"迭代次数：{i}")
print(f"参数w0，w1，w2的值：{w0},{w1},{w2}")