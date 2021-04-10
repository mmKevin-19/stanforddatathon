from sklearn import linear_model
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plot

data = datasets.load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df
y = target["MEDV"]

lm = linear_model.LinearRegression()
model = lm.fit(X, y)

predictions = lm.predict(X)
print(predictions[0:5])
# R^2
print(lm.score(X, y))
print(lm.coef_)
print(lm.intercept_)

# plot.scatter(X, y, color='blue')
# plot.plot(X, )
