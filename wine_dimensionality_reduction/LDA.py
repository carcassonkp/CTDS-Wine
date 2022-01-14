from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from winedata import import_winedata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
features, labels = import_winedata()
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

accuracy = {}
for i in range(len(features.columns), 0, -1):
    if i == 11:
        # RANDOM FOREST BEFORE DIMENSIONALITY REDUCTION
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                                    stratify=labels, random_state=0)
        # ###########################
        model = RandomForestClassifier(random_state=0)
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)

        # ###########################
        report = classification_report(test_labels, predictions, zero_division=1, output_dict=True)
        # # print(report)
        # accuracy.append(report['accuracy'])
        # print(accuracy)
        accuracy[i] = report['accuracy']
    else:
        if(i<min(len(features.columns), 6-1)): # ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
            print("i:{}".format(i))
            print("LDA for {} components".format(i))
            lda = LinearDiscriminantAnalysis(n_components=i)
            features_lda = lda.fit(features, labels).transform(features)
            train_features, test_features, train_labels, test_labels = train_test_split(features_lda, labels, test_size=0.2,
                                                                                        stratify=labels, random_state=0)
            model = RandomForestClassifier(random_state=0)
            model.fit(train_features, train_labels)
            predictions = model.predict(test_features)
            report = classification_report(test_labels, predictions, zero_division=1, output_dict=True)
            accuracy[i] = report['accuracy']

print(accuracy)

x=accuracy.keys()
y=accuracy.values()

plt.scatter(x, y)
plt.xlabel('Dimensions(LDA)')
plt.ylabel('Accuracy Values')
plt.show()
