import numpy as np
#定义批量梯度下降算法
def bgd_optimizer(target_fn, grad_fn, init_W, X, Y, lr=0.0001, tolerance = 1e-12, max_iter=1000000):
    W = init_W
    target_value=target_fn(W, X, Y)
    for i in range(max_iter):
        grad = grad_fn(W, X, Y)
        next_W = W - grad * lr
        next_target_value = target_fn(next_W, X, Y)
        if abs(next_target_value-target_value) < tolerance:
            return i, next_W
        else:
            W,  target_value = next_W, next_target_value
    return i, None

#定义随机梯度下降算法
def sgd_optimizer(target_fn, grad_fn, init_W, X, Y, lr = 0.0001,tolerance=1e-12, max_iter=10000000):
    W, rate = init_W, lr
    min_W, min_target_value = None, float("inf")
    no_improvement = 0
    target_value = target_fn(W, X, Y)
    for i in range(max_iter):
        index = np.random.randint(0,len(X))
        gradient = grad_fn(W, X[index], Y[index])
        W = W - lr * gradient
        new_target_value = target_fn(W, X, Y)
        if abs(new_target_value-target_value)<tolerance:
            return i, W
        target_value = new_target_value
    return i, None

#定义目标函数
def target_function(W, X, Y):
    w0,w1 = W
    return np.sum((w0+X*w1-Y)**2) / (2*len(X))

#计算梯度
def grad_function(W, X, Y):
    w0, w1 = W
    w0_grad = np.sum(w0+X*w1-Y) / len(X)
    w1_grad = X.dot(w0+w1-Y) / len(X)
    return np.array([w0_grad, w1_grad])

#训练数据集
x = np.array([75, 87, 105, 110, 120], dtype=np.float64)
y = np.array([270, 280, 295, 310, 335], dtype=np.float64)

#随机化初始化权重参数
np.random.seed(0)
init_W = np.array([np.random.random(), np.random.random()])

#梯度下降求解
i, W = bgd_optimizer(target_function, grad_function, init_W, x, y)
if W is not None:
    w0, w1 = W
    print(f"迭代次数：{i},最优的w0和w1:({w0},{w1})")