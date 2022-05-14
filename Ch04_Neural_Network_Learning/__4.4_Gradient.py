
import numpy as np


def function_2(x):
    return np.sum(x**2)


def partial_differentiation(f, x, i):
    tmp1 = x.copy()
    tmp2 = x.copy()
    h = 1e-4
    tmp1[i] += h
    tmp2[i] -= h
    return (f(tmp1) - f(tmp2)) / (2*h)


def numerical_gradient(f, x):
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


print(partial_differentiation(function_2, np.array([3.0, 4.0]), 0))
print(numerical_gradient(function_2, np.array([3.0, 4.0])))
