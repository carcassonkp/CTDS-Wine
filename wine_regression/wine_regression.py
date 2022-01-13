import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor

from sklearn import metrics
df = pd.read_csv('winequality/winequality-red.csv', sep=';')

if (df.isnull().values.any()):
    print("Missing Values Found")  # check for missing values
else:
    print("No Missing Values Found")

X = df.drop(columns='quality')
Y = df['quality']

train_X, test_X, train_Y, test_Y = train_test_split(train, Y, test_size=0.2, stratify=Y, random_state=88)

####################
# model = LinearSVR(max_iter=20000)
# model = NuSVR()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = AdaBoostRegressor()
# model = MLPRegressor()
model = SGDRegressor()

##########################
model.fit(train_X, train_Y)
predict_Y = model.predict(test_X)
###########################

mse = metrics.mean_squared_error(test_Y, predict_Y)
print("Mean Squared Error:", mse)

mae = metrics.mean_absolute_error(test_Y, predict_Y)
print("Mean Absolute Error:", mae)

mape = metrics.mean_absolute_percentage_error(test_Y, predict_Y)
print("Mean Absolute Percentage Error:", mape)

mdae = metrics.median_absolute_error(test_Y, predict_Y)
print("Median Absolute Error:", mdae)