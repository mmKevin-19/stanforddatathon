from sklearn import linear_model
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plot

# all columns
allCols = pd.read_csv("data/lifestyle.csv")
allCols = allCols.iloc[:, 1:-3]
print(allCols.columns)
# for s in allCols.columns:
#     print(s)

models = []
numCols = len(allCols.columns)
for i in range(numCols):
    X = allCols.iloc[:, i]
    # X = allCols[1]
    y = pd.read_csv("data/lifestyle.csv", usecols=[2])
    y = y.astype('int')

    lm = linear_model.LinearRegression()
    model = lm.fit(X.reshape(-1, 1), y)
    models.append({
        'coef': lm.coef_,
        'intercept': lm.intercept_,
        'score': lm.score
    })
    plot.scatter(X, y, color='red')
    plot.plot(X, lm.predict(X), color='blue')
    plot.show()

# sortedmodels = sorted(models, key=lambda k: abs(k['coef']))
# predictions = lm.predict(X)
# print(predictions[0:5])

# print("score: ", lm.score(X, y))
# print("coef: ", lm.coef_)
# print("intercept: ", lm.intercept_)
# print(sortedmodels)
