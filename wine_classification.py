import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

df = pd.read_csv('winequality/winequality-red.csv', sep=';')

if (df.isnull().values.any()):
    print("Missing Values Found")  # check for missing values
else:
    print("No Missing Values Found")

X = df.drop(columns='quality')
Y = df['quality']

# fig = px.histogram(Y, x='quality')
# fig.show()
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=88)

# fig = px.histogram(train_Y, x='quality')
# fig.show()
# fig = px.histogram(test_Y, x='quality')
# fig.show()

###########################
# model = LinearSVC(max_iter=20000, dual=False) #0.56 accuracy
# model = KNeighborsClassifier()# 0.50 accuracy
# model = DecisionTreeClassifier()  # 0.60 accuracy
# model = RandomForestClassifier() # 0.71 accuracy
# model = AdaBoostClassifier()  # 0.51 accuracy
model = MLPClassifier()  # 0.59 accuracy

model.fit(train_X, train_Y)
predict_Y = model.predict(test_X)
###########################
print(classification_report(test_Y, predict_Y))

mse = metrics.mean_squared_error(test_Y, predict_Y)
print("Mean Squared Error:", mse)

mae = metrics.mean_absolute_error(test_Y, predict_Y)
print("Mean Absolute Error:", mae)

mape = metrics.mean_absolute_percentage_error(test_Y, predict_Y)
print("Mean Absolute Percentage Error:", mape)

mdae = metrics.median_absolute_error(test_Y, predict_Y)
print("Median Absolute Error:", mdae)