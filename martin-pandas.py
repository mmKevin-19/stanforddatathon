from sklearn import linear_model
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plot

# named df not df_csv
X = pd.read_csv("lifestyle.csv", usecols=[1, 3, 4])
y = pd.read_csv("lifestyle.csv", usecols=[2])
y = y.astype('int')

lm = linear_model.LinearRegression()
model = lm.fit(X, y)

predictions = lm.predict(X)
print(predictions[0:5])

print("score: ", lm.score(X, y))
print("coef: ", lm.coef_)
print("intercept: ", lm.intercept_)
