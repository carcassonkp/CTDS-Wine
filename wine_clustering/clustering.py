from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from winedata import import_winedata
import sys

sys.stdout = open('logs/accuracies.txt', 'wt')

features, labels = import_winedata()

# plt.hist(labels.values)
# plt.show()
k_labels = KMeans(n_clusters=6, random_state=1, max_iter=1000).fit_predict(features)
meanshift_labels = MeanShift().fit_predict(features)
gaussianmixture_labels = GaussianMixture(n_components=6).fit_predict(features)


plt.subplot(4, 1, 1)
plt.scatter(features.iloc[:, 0], labels, c=meanshift_labels,s=5)
plt.title("MeanShift")
plt.subplot(4,1,2)
plt.scatter(features.iloc[:,0], labels, c=k_labels, s=5)
plt.title("K_means")
plt.subplot(4,1,3)
plt.scatter(features.iloc[:,0], labels, c=labels, s=5)
plt.title("Ground Truth")
plt.subplot(4,1,4)
plt.scatter(features.iloc[:,0], labels, c=gaussianmixture_labels, s=5)
plt.title("Gaussian Mixture")
plt.show()

print("K_ means accuracy")
print(classification_report(labels, k_labels, zero_division=1))
print("Mean shift accuracy")
print(classification_report(labels, meanshift_labels, zero_division=1))
print("Gaussian Mixture accuracy")
print(classification_report(labels, gaussianmixture_labels, zero_division=1))