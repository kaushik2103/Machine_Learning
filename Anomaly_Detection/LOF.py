# Anomoaly Decetion using Local Outlier Factor Algorithm

from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

X, Y = make_circles(n_samples=750, factor=0.3, noise=0.1)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof.fit_predict(X)
print("Negative Outlier Factor: ", lof.negative_outlier_factor_)
plt.scatter(X[:, 0], X[:, 1], c=lof.negative_outlier_factor_)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()
