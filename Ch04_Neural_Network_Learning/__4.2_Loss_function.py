
import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
ans1 = sum_squares_error(np.array(y1), np.array(t))
ans2 = sum_squares_error(np.array(y2), np.array(t))
print("sum_squares_error(y1): ", ans1)
print("sum_squares_error(y2): ", ans2)


def cross_entropy_error2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# t = [0, 0, 1, 0, 0, ...] => t = [2]
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


ans3 = cross_entropy_error2(np.array(y1), np.array([2]))
ans4 = cross_entropy_error2(np.array(y2), np.array([2]))
print("cross_entropy_error(y1): ", ans3)
print("cross_entropy_error(y2): ", ans4)


# 4.2.3 mini batch practice
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(train_size)
print(x_batch.shape)
print(batch_mask)

