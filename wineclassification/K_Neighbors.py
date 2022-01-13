from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from winedata import import_winedata
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import sys

sys.stdout = open('logs/K_Neighbors.txt', 'wt')

features, labels = import_winedata()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            stratify=labels, random_state=0)
# ###########################
# Find best k for model
k_range = range(1, 50)
# Number of k from 1 to 50
k_scores = []
k_best_score = 0

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)  # It’s 10 fold cross validation with ‘accuracy’ scoring
    scores = cross_val_score(knn, features, labels, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    if scores.mean() > k_best_score:
        k_best = k
        k_best_score = scores.mean()

# print(k_best)
# print(k_best_score)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()
# ###########################
#
model = KNeighborsClassifier(n_neighbors=k_best)
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
