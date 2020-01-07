from sklearn.datasets.samples_generator import make_blobs, make_circles
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxopt
import numpy as np
plt.style.use('seaborn-whitegrid')
random.seed(1337)


class SVM:
    KERNEL_DICT = {
        'linear': lambda x, y: np.dot(x, y),
        # GAUSSIAN RBF
        'gaussian-rbf': lambda x, y, s=1: np.exp(-np.linalg.norm(x - y)**2 / (2 * (s ** 2))),
        # POLYNOMIAL RBF
        'polynomial-rbf': lambda x, y, p: (np.dot(x, y) + 1) ** p
    }

    def __init__(self, kernel=None, C=None):
        if kernel is None or kernel == 'linear':
            self.kernel = self.KERNEL_DICT['linear']
        else:
            self.kernel = self.KERNEL_DICT[kernel]
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        # this represents the ti*tj*(xi_T . xj)
        H = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                #H[i][j] = np.dot(X[i], X[j]) * y[i] * y[j]
                H[i][j] = self.kernel(X[i], X[j]) * y[i] * y[j]

        # this is the coefficient matrix of -sum(alpha_i)
        f = -np.ones(len(X))

        cv_H = cvxopt.matrix(H.astype(float))
        cv_f = cvxopt.matrix(f.astype(float))
        # constraint: a_i >= 0
        cv_G = cvxopt.matrix(-np.eye(len(X)))
        cv_h = cvxopt.matrix(np.zeros(len(X)))
        # constraint: alpha_i * y_i = 0 -> Ax = 0
        cv_A = cvxopt.matrix(y.reshape(1, -1).astype(float))
        cv_b = cvxopt.matrix(np.zeros(1))

        sol = cvxopt.solvers.qp(cv_H, cv_f, cv_G, cv_h, cv_A, cv_b)

        self.alpha = np.array(sol['x'])

        # Selecting the set of indices S corresponding to non zero parameters
        self.S = (self.alpha > 1e-4).flatten()
        ind = np.arange(len(self.alpha))[self.S]

        self.alpha_filtered = self.alpha[self.S]

        # w parameter in vectorized form
        # @: matrix multiplication
        self.w = (np.matmul((y * self.alpha).T, X)).reshape(-1, 1)

        # Computing b if kernel is linear
        self.b = y[self.S] - np.dot(X[self.S], self.w)

    def get_results(self):
        # Unpack this function into alpha, w, S, b
        return self.alpha, self.w, self.S, self.b

    def discriminant(self, alpha, w0, X, t, x):
        """
        Parameters
        ----------
        alpha: list
        w0: float
        X: list of lists
        t: list
        x: list of lists

        Returns
        -------
        arr: list
        """

        # filters the input array, since support vectors have nonzero lagrange multipliers
        alpha_dict = {}
        for ind, a in enumerate(alpha):
            if a > 1e-4:
                alpha_dict[ind] = a

        arr = []
        if self.kernel == self.KERNEL_DICT['linear']:  # if kernel is None -> linear kernel
            for i in x:
                s = 0
                for key in alpha_dict:
                    s += alpha_dict[key] * t[key] * np.dot(X[key].T, i)
                s += w0
                arr.append(s)
        else:
            # TODO: implement kernel check
            pass
        return arr

    def decision_function(self, alpha, X, t, x):
        alpha_dict = {}
        for ind, a in enumerate(alpha):
            if a > 1e-4:
                alpha_dict[ind] = a

        arr = []
        for i in x:
            d = 0
            for key in alpha_dict:
                d += alpha_dict[key] * y[key] * self.kernel(X[key], i)
            arr.append(d)

        return arr

    def calc_contour(self, X, t):
        x_space = np.linspace(-20, 10, 50)
        y_space = np.linspace(-20, 10, 50)

        X_coords, Y_coords = np.meshgrid(x_space, y_space)
        X_new = X_coords.ravel()
        Y_new = Y_coords.ravel()

        #Z = []
        # for i in range(len(X_new)):
        #    Z.append(list(self.discriminant(self.alpha, self.b[0], X, t, [X_new[i], Y_new[i]])))
        if self.kernel == self.KERNEL_DICT['linear']:
            Z = self.discriminant(self.alpha, self.b[0], X, t, zip(X_new, Y_new))
        else:
            Z = self.decision_function(self.alpha, X, t, zip(X_new, Y_new))
        return X_coords, Y_coords, np.array(Z).ravel().reshape(X_coords.shape)


def plot_svm(w, b, X, y):
    def f(x, w, b, c=0):
        return (-w[0] * x - b + c) / w[1]

    # comment out the ax.plot to display on a separate plot

    #fig = plt.figure(figsize=(8, 8))
    a0 = -15.0
    a1 = f(a0, w, b)
    b0 = 4
    b1 = f(b0, w, b)
    #plt.plot([a0, b0], [a1[0], b1[0]], "k")
    ax.plot([a0, b0], [a1[0], b1[0]], "k")

    # w.x + b = 1
    a0 = -15.0
    a1 = f(a0, w, b, 1)
    b0 = 4
    b1 = f(b0, w, b, 1)
    #plt.plot([a0, b0], [a1[0], b1[0]], "k--")
    ax.plot([a0, b0], [a1[0], b1[0]], "k--")

    # w.x + b = -1
    a0 = -15.0
    a1 = f(a0, w, b, -1)
    b0 = 4
    b1 = f(b0, w, b, -1)
    #plt.plot([a0, b0], [a1[0], b1[0]], "k--")
    ax.plot([a0, b0], [a1[0], b1[0]], "k--")

    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    # plt.show()


if __name__ == '__main__':
    # generate two linearly separable culsters of data
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=1, shuffle=True, cluster_std=2)

    y[y == 0] = -1
    y_orig = y

    y = y.reshape(-1, 1).astype(float)

    svm = SVM()
    svm.fit(X, y)
    a, w, S, b = svm.get_results()
    print(f'Alpha: {a[a > 1e-4]} \nw: {w.flatten()}\nb: {b[0]}')

    #random_point, _ = make_blobs(n_samples=10, centers=2, n_features=2, random_state=1, shuffle=True, cluster_std=1.7)
    # res = svm.discriminant(a, b[0], X, y, [-1., 0.])
    # print(res)

    # LINEAR
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    A, B, C = svm.calc_contour(X, y)
    ax.plot_surface(A, B, C, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(X[:, 0], X[:, 1], zs=0, zdir='z', c=y_orig, cmap='winter')
    plot_svm(w, b[0], X, y_orig)
    plt.show()

    # RBF 3D plot
    rbf = SVM(kernel='gaussian-rbf')
    rbf.fit(X, y)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    A, B, C = rbf.calc_contour(X, y)
    ax.plot_surface(A, B, C, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], zs=0, zdir='z', c=y_orig, cmap='winter')
    plt.show()

    # RBF 2d plot
    rbf = SVM(kernel='gaussian-rbf')
    rbf.fit(X, y)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()
    A, B, C = rbf.calc_contour(X, y)
    ax.contour(A, B, C)
    ax.scatter(X[:, 0], X[:, 1], c=y_orig, cmap='winter')
    plt.show()
