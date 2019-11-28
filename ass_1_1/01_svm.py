from sklearn.datasets.samples_generator import make_blobs
import random
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
random.seed(1337)

# generate two linearly separable culsters of data
X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
plt.show()
