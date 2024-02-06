# DBSCAN Clustering

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, Y = make_moons(n_samples=250, noise=0.05)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN Algorithm
dbscan = DBSCAN(eps=0.5)
dbscan.fit(X_scaled)
print("DBSCAN Clustering Labels: ", dbscan.labels_)

plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()
