
import sys, os
import numpy as np
import pickle

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

y = np.array([[0.1, 0.2, 0.7], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])
t = np.array([2, 1, 0])
# a = -np.sum(np.log(y[np.arange(3), t] + 1e-7)) / 3
print(np.log(y[np.arange(3),t]))