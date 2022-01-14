import matplotlib.pyplot as plt

r2 = [0.43, -0.02, 0.18, 0.33, 0.34, 0.52, 0]
regression = ['Ada', 'DT', 'KN', 'SVR', 'MLP', 'RF', 'SGD']

plt.scatter(regression, r2)
plt.xlabel('Type of Regression')
plt.ylabel('R2 Score')
plt.show()