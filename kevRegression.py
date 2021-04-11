import numpy   
import matplotlib.pyplot as plot
import pandas 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pandas.read_csv('dartdata.csv')
x = dataset.iloc[:, 2]
y = dataset.iloc[:, 1]

x = x.values.reshape(-1, 1)

linearRegressor = LinearRegression()

linearRegressor.fit(x, y)

plot.scatter(x, y, color='red', alpha=.1)
plot.plot(x, linearRegressor.predict(x), color='blue')
plot.xlabel("Hours of Sleep")
plot.ylabel("Stress (measured from a scale 1 - 7 in increasing stress)")
plot.title("The correlation between sleep and stress levels")
plot.show()