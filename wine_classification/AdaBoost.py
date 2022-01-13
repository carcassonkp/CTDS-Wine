from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from winedata import import_winedata
import matplotlib.pyplot as plt
import sys

sys.stdout = open('logs/Ada_Boost.txt', 'wt')

features, labels = import_winedata()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            stratify=labels, random_state=0)
# ###########################
model = AdaBoostClassifier()
model.fit(train_features, train_labels)
predictions = model.predict(test_features)

# ###########################
print(classification_report(test_labels, predictions, zero_division=1))
#
mse = metrics.mean_squared_error(test_labels, predictions)
print("Mean Squared Error:", mse)

mae = metrics.mean_absolute_error(test_labels, predictions)
print("Mean Absolute Error:", mae)

mape = metrics.mean_absolute_percentage_error(test_labels, predictions)
print("Mean Absolute Percentage Error:", mape)

mdae = metrics.median_absolute_error(test_labels, predictions)
print("Median Absolute Error:", mdae)
