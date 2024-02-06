import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris.target, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Scatter plot
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.show()

# Agglomerative Clustering
import scipy.cluster.hierarchy as shc

plt.Figure(figsize=(10, 7))
plt.title("Dendrograms")
shc.dendrogram(shc.linkage(X_train_pca, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Data points")
plt.ylabel("Euclidean distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, linkage="ward")
cluster.fit(X_train_pca)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster.labels_)
plt.show()

from sklearn.metrics import silhouette_score

silhouette_coeff = []
for k in range(2, 10):
    cluster = AgglomerativeClustering(n_clusters=k, linkage="ward")
    cluster.fit(X_train_pca)
    silhouette_coeff.append(silhouette_score(X_train_pca, cluster.labels_))

plt.plot(range(2, 10), silhouette_coeff)
plt.xticks(range(2, 10))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()