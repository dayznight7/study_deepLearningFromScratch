
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


def function_2(x):
    return np.sum(x**2)


def my_partial_differentiation(f, x, i):
    tmp1 = x.copy()
    tmp2 = x.copy()
    h = 1e-4
    tmp1[i] += h
    tmp2[i] -= h
    return (f(tmp1) - f(tmp2)) / (2*h)


def my_numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


print(my_partial_differentiation(function_2, np.array([3.0, 4.0]), 0))
print(numerical_gradient(function_2, np.array([3.0, 4.0])))


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))


class simpleNet:

    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])
print("x:", x)
net = simpleNet()
print("W:", net.W)
p = net.predict(x)
print("p = x * W:", p)

print("index of max result:", np.argmax(p))

#answer
t = np.array([0, 0, 1])
print("net.loss(x, t):", net.loss(x, t))

def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
