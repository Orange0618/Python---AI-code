import numpy as np
xTrain = np.array([75,87,105,110,120])
yTrain = np.array([270,280,295,310,335])
w1 = np.cov(xTrain,yTrain,ddof = 1)[1,0] / np.var(xTrain, ddof=1)
w0 = np.mean(yTrain)-w1*np.mean(xTrain)
print(f"w1={w1}")
print(f"w0={w0}")