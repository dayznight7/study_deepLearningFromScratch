
import numpy as np
import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(x_train[0].shape)
print(x_train.shape[0])

a = [[1, 2], [3, 4], [5, 6]]
a = np.array(a)
print(a.shape)
print(a.shape[0])
print(a[0])