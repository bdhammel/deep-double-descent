import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

N = 1000
N_test = 100
x = np.linspace(0, 1, N)
y = 2.5*x + np.random.normal(0, .1, size=N)

train_idx = np.random.choice(range(1, N-1), replace=False, size=N_test)

train_x = x[train_idx].reshape(-1, 1)
train_y = y[train_idx]
test_x = x.reshape(-1, 1)

prams = 500
train_X = np.hstack([train_x**n for n in range(1, prams)])
test_X = np.hstack([test_x**n for n in range(1, prams)])

reg = LinearRegression().fit(train_X, train_y)
p = reg.predict(test_X)

plt.plot(x, y, '.', alpha=.3)
plt.plot(x[train_idx], y[train_idx], 'o', alpha=.3)
plt.plot(x, p)
plt.ylim(0, 2.5)
