import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

x = np.linspace(-5.0, 5.0, 100)
y = np.sqrt(10 ** 2 - x ** 2)
y = np.hstack([y, -y])
x = np.hstack([x, -x])
x1 = np.linspace(-5.0, 5.0, 100)
y1 = np.sqrt(5 ** 2 - x1 ** 2)
y1 = np.hstack([y1, -y1])
x1 = np.hstack([x1, -x1])
plt.scatter(y, x)
plt.scatter(y1, x1)

plt.show()

import pandas as pd

df1 = pd.DataFrame(np.vstack([y, x]).T, columns=['X1', 'X2'])
df1['Y'] = 0
df2 = pd.DataFrame(np.vstack([y1, x1]).T, columns=['X1', 'X2'])
df2['Y'] = 1
df = df1._append(df2)
df.head(5)

# split data into X and y
X = df.iloc[:, :2]
y = df.Y

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

classifier = SVC(kernel='linear')
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))

df['X1_Square'] = df['X1'] ** 2
df['X2_Square'] = df['X2'] ** 2
df['X1*X2'] = (df['X1'] * df['X2'])

X = df[['X1', 'X2', 'X1_Square', 'X2_Square', 'X1*X2']]
Y = df['Y']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

import plotly.express as px

fig = px.scatter_3d(df, x='X1', y='X2', z='X1*X2', color='Y')
fig.show()

fig2 = px.scatter_3d(df, x='X1_Square', y='X2_Square', z='X1*X2', color='Y')
fig2.show()

classifier = SVC(kernel='linear')
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))

