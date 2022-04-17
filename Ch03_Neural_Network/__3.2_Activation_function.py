
import numpy as np
import matplotlib.pylab as plt

def sigmod(x):
    return 1 / (1 + np.exp(-x))


# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmod(x)
# plt.plot(x, y)
# plt.show()


def step_function(x):
    y = x > 0
    return y.astype(int)


# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.show()


def relu(x):
    return np.maximum(0, x)


