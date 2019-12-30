from sklearn.datasets.samples_generator import make_blobs
import random
import matplotlib.pyplot as plt
import cvxopt
import numpy as np
plt.style.use('seaborn-whitegrid')
random.seed(1337)

# generate two linearly separable culsters of data
X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=1, shuffle=True, cluster_std=1.7)

y[y == 0] = -1
y_orig = y

y = y.reshape(-1, 1).astype(float)

# this represents the ti*tj*(xi_T . xj)
H = np.zeros((len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        H[i][j] = np.dot(X[i], X[j]) * y[i] * y[j]

# this is the coefficient matrix of -sum(alpha_i)
f = -np.ones(len(X))

cv_H = cvxopt.matrix(H.astype(float))
cv_f = cvxopt.matrix(f.astype(float))
# constraint: (w_T . xi + b)*yi >= 1 or dummy constraint????? idk
cv_G = cvxopt.matrix(-np.eye(len(X)))
cv_h = cvxopt.matrix(np.zeros(len(X)))
# constraint: alpha_i * y_i = 0 -> Ax = 0
cv_A = cvxopt.matrix(y.reshape(1, -1).astype(float))
cv_b = cvxopt.matrix(np.zeros(1))

sol = cvxopt.solvers.qp(cv_H, cv_f, cv_G, cv_h, cv_A, cv_b)
alpha = np.array(sol['x'])

# w parameter in vectorized form
# @: matrix multiplication
w = (np.matmul((y * alpha).T, X)).reshape(-1, 1)

# Selecting the set of indices S corresponding to non zero parameters
S = (alpha > 1e-4).flatten()

# Computing b
b = y[S] - np.dot(X[S], w)

# Display results
print('Alphas = ', alpha[alpha > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])


def f(x, w, b, c=0):
    return (-w[0] * x - b + c) / w[1]


fig = plt.figure(figsize=(8, 8))
a0 = -15.0
a1 = f(a0, w, b)
b0 = 4
b1 = f(b0, w, b)
plt.plot([a0, b0], [a1[0], b1[0]], "k")

# w.x + b = 1
a0 = -15.0
a1 = f(a0, w, b, 1)
b0 = 4
b1 = f(b0, w, b, 1)
plt.plot([a0, b0], [a1[0], b1[0]], "k--")

# w.x + b = -1
a0 = -15.0
a1 = f(a0, w, b, -1)
b0 = 4
b1 = f(b0, w, b, -1)
plt.plot([a0, b0], [a1[0], b1[0]], "k--")

plt.scatter(X[:, 0], X[:, 1], c=y_orig, cmap='winter')
plt.show()
