import matplotlib.pyplot as plt

accuracy = [0.53, 0.59, 0.60, 0.71, 0.65, 0.60]
classification = ['KN', 'SVC', 'MLP', 'RF', 'RF (SMOTE)', 'DT']

plt.scatter(classification, accuracy)
plt.xlabel('Type of Classification')
plt.ylabel('Classification accuracy')
plt.show()
