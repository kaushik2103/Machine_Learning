# Anomaly Detection using Isolation Forest Algorithm

import pandas as pd
df = pd.read_csv('heart.csv')

import matplotlib.pyplot as plt
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.show()

from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination='auto')
clf.fit(df)
predictions = clf.predict(df)

print("Predictions: ", predictions)

import numpy as np
index = np.where(predictions < 0)

print("Index: ", index)

x = df.values
index = np.where(predictions < 0)
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.scatter(x[index, 0], x[index, 1], color='r')
plt.show()

