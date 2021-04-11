from sklearn import linear_model
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

# all columns
allCols = pd.read_csv("data/lifestyle.csv")
allCols = allCols.iloc[:200, 1:-3]
print('here', allCols.columns)
# for s in allCols.columns:
#     print(s)

models = []
numCols = len(allCols.columns)
for i in range(numCols):
    # y = pd.read_csv("data/lifestyle.csv", usecols=[2], nrows=100)
    # y = y.astype('int')

    # stress levels
    y = allCols.iloc[:, 1]
    print(y)
    X = allCols.iloc[:, i]
    # X = X.values.reshape(-1, 1)
    corrCoef = np.corrcoef(X, y)
    print('corrCoef: ', corrCoef)

    X = X.values.reshape(-1, 1)

    lm = linear_model.LinearRegression()
    model = lm.fit(X, y)
    models.append({
        'coef': lm.coef_,
        'intercept': lm.intercept_,
        'score': lm.score
    })
    plot.scatter(X, y, color='red', alpha=.1)
    plot.plot(X, lm.predict(X), color='blue')
    plot.ylabel('Stress Level (Rated 1-5)')
    plot.xlabel(allCols.columns[i])
    plot.title('Effect of ___ on ___')
    plot.show()

# sortedmodels = sorted(models, key=lambda k: abs(k['coef']))
# predictions = lm.predict(X)
# print(predictions[0:5])

# print("score: ", lm.score(X, y))
# print("coef: ", lm.coef_)
# print("intercept: ", lm.intercept_)
# print(sortedmodels)
