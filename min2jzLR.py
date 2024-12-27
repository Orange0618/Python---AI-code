import numpy as np
def linreg_matrix(x,y):
    X_X_T=np.matmul(x,x.T)
    X_X_T_1 = np.linalg.inv(X_X_T)
    X_X_T_1_X = np.matmul(X_X_T_1,x)
    X_X_T_1_X_Y = np.matmul(X_X_T_1_X,y)
    return X_X_T_1_X_Y
xTrain = np.array([[75],[87],[105],[110],[120]])
yTrian = np.array([270,280,295,310,335])
#对x进行扩展，加入一个全1的行
def make_ext(x):
    ones = np.ones([1,np.size(xTrain)])
    new_x = np.insert(x,0,ones,axis=0)
    return new_x

x = make_ext(xTrain.T)
y = yTrian

print(f"x={x}")
print(f"y={y}")

w = linreg_matrix(x,y)
print(f"w={w}")