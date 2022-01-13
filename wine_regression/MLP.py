
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from winedata import import_winedata
from sklearn import metrics
import sys

sys.stdout = open('logs/MLP.txt', 'wt')

features, labels = import_winedata()
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            stratify=labels, random_state=0)
####################
model = MLPRegressor()
model.fit(train_features, train_labels)
predict_Y = model.predict(test_features)

##########################
mse = metrics.mean_squared_error(test_labels, predict_Y)
print("Mean Squared Error:", mse)

mae = metrics.mean_absolute_error(test_labels, predict_Y)
print("Mean Absolute Error:", mae)

mape = metrics.mean_absolute_percentage_error(test_labels, predict_Y)
print("Mean Absolute Percentage Error:", mape)

mdae = metrics.median_absolute_error(test_labels, predict_Y)
print("Median Absolute Error:", mdae)

r2 = metrics.r2_score(test_labels, predict_Y)
print("R2 Score :", r2)