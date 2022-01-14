import matplotlib.pyplot as plt

mse = [0.367, 0.65625, 0.5313609467455621, 0.431368558620853, 0.42738759236066437, 0.31211218749999997]
regression = ['Ada', 'DT', 'KN', 'SVR', 'MLP', 'RF']

plt.scatter(regression, mse)
plt.xlabel('Type of Regression')
plt.ylabel('MSE')
plt.show()