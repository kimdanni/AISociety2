import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Generate random data
np.random.seed(0)
X1 = np.linspace(0, 100, 100)
X2 = np.linspace(50, 0, 100)
Y = 2*X1 + 3*X2 + 5*np.random.randn(100)

# Create a pandas DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# Create a linear regression model
model = LinearRegression()
X = data[['X1', 'X2']]
model.fit(X, Y)

# Create a 3D scatter plot of the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['X1'], data['X2'], data['Y'], c='red', marker='o')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# Plot the regression plane
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
X1, X2 = np.meshgrid(np.linspace(x_min, x_max, 10),
                     np.linspace(y_min, y_max, 10))
Y = model.intercept_ + model.coef_[0]*X1 + model.coef_[1]*X2
ax.plot_surface(X1, X2, Y, alpha=0.2)

plt.show()
