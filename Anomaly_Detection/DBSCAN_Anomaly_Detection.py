# Anomoaly Decetion using DBSCAN Algorithm

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

X, Y = make_circles(n_samples=750, factor=0.3, noise=0.1)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

dbscan = DBSCAN(eps=0.10, min_samples=5)
dbscan.fit_predict(X)
print("Labels: ", dbscan.labels_)

plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()