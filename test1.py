import numpy as np
from sklearn.linear_model import LinearRegression

#训练数据集
xTrain = np.array([[75],[87],[105],[110],[120]])
yTrain = np.array([270, 280, 295, 310, 335])
xTest = np.array([[85], [90], [93], [109]])
yTest = np.array([280,282,284,305])
#建立模型并求解
model = LinearRegression()
model.fit(xTrain, yTrain)

#输出求解结果
print(f"b={model.intercept_}")
print(f"w1={model.coef_}")

#使用模型进行预测
yTrainPredicted = model.predict(xTrain) #训练集
yTestPredicted = model.predict(xTest)   #测试集

#模型评价
#残差
ssResTrain = np.sum((yTrainPredicted-yTrain)**2)
print(f"手动训练残差={ssResTrain}")
#print(f"自动训练残差={model._residues}")

#R方
#手动计算测试R方
ssResTest = np.sum((yTestPredicted-yTest)**2)
ssTotalTest = np.sum((np.mean(yTest)-yTest)**2)
rsquareTest = 1 - ssResTest / ssTotalTest
print(f"手动测试R方={rsquareTest}")
#自动计算测试R方
print(f"自动测试R方={model.score(xTest,yTest)}")