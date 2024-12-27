import numpy as np
from sklearn.linear_model import LogisticRegression
xTrain = np.array([[94], [120], [160], [170], [155], [78], [75], [80], [100], [125]])
yTrain = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression(solver="lbfgs")
model.fit(xTrain, yTrain)

newX = np.array([[100], [130]])
newY = model.predict(newX)

print(f"newX:{newX}")
print(f"newY:{newY}")
print(f"model.coef_:{model.coef_}")
print(f"model.intercept_:{model.intercept_}")
