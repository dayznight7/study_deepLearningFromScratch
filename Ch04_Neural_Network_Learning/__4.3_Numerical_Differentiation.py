
import numpy as np

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 4.3.3 partial differentiation
# f(x0,x1) = x0^2 + x1^2


def function_2(x):
    return np.sum(x**2)


# p126 Q1: x0=3, x1=4, df/dx0 = ?
print(numerical_diff(function_2, 3.0))


def partial_differentiation(f, x, i):
    tmp1 = x.copy()
    tmp2 = x.copy()
    h = 1e-4
    tmp1[i] += h
    tmp2[i] -= h
    return (f(tmp1) - f(tmp2)) / (2*h)


# df/dx1
print(partial_differentiation(function_2, np.array([3.0, 4.0]), 1))
