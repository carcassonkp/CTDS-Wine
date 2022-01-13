from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from winedata import import_winedata

features, labels = import_winedata()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            stratify=labels, random_state=88)

###########################
# model = LinearSVC(max_iter=20000, dual=False) #0.56 accuracy
# model = KNeighborsClassifier()# 0.50 accuracy
# model = DecisionTreeClassifier()  # 0.60 accuracy
# model = RandomForestClassifier() # 0.71 accuracy
# model = AdaBoostClassifier()  # 0.51 accuracy
model = MLPClassifier()  # 0.59 accuracy

model.fit(train_features, train_labels)
predictions = model.predict(test_features)
###########################
print(classification_report(test_labels, predictions))

mse = metrics.mean_squared_error(test_labels, predictions)
print("Mean Squared Error:", mse)

mae = metrics.mean_absolute_error(test_labels, predictions)
print("Mean Absolute Error:", mae)

mape = metrics.mean_absolute_percentage_error(test_labels, predictions)
print("Mean Absolute Percentage Error:", mape)

mdae = metrics.median_absolute_error(test_labels, predictions)
print("Median Absolute Error:", mdae)
