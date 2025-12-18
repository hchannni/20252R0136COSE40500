import numpy as np
from scipy.optimize import minimize


def MSE(x, A, b):
    residual = A @ x - b
    mse = np.mean(residual ** 2)
    return mse


m = 1000
n = 10

mu = 0
sig = 1

mu_noise = 0
sig_noise = 0.1

x_true = np.random.normal(mu, sig, n)

A = np.c_[np.ones((m, 1)), np.random.normal(mu, sig, (m, n - 1))]
b = A @ x_true + np.random.normal(mu_noise, sig_noise, m)

x0 = np.random.normal(mu, sig, n)

estim = minimize(MSE, x0, args=(A, b))

print('solution from minimize:', estim.x)
print('true x', x_true)

print('error percentage:', np.linalg.norm(x_true - estim.x) / np.linalg.norm(x_true) * 100, '%')

