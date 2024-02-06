# K - Means Clustering Algorithm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# For Elbow Method Automated
from kneed import KneeLocator

X, Y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=23)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Manually Processing
# Elbow Method to select K value
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Making Model
kmeans = KMeans(n_clusters=3, init='k-means++')
y_labels = kmeans.fit_predict(X_train)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_labels)
plt.show()

y_test = kmeans.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.show()

# Automated Elbow Method
kneed_locator = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
print("Optimal K value is: ", kneed_locator.elbow)

# Performance Metrics
print("Silhouette Score: ")
silhouette_coeff = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_train)
    score = silhouette_score(X_train, kmeans.labels_)
    silhouette_coeff.append(score)

print("Silhouette Score: ", silhouette_coeff)

plt.plot(range(2, 11), silhouette_coeff)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
