# Linear Regression

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading data
df = fetch_california_housing()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names

# Independent and dependent variables
X = dataset
Y = df.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Standardization of data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression Model

regression = LinearRegression()
regression.fit(X_train, Y_train)
mse = cross_val_score(regression, X_train, Y_train, scoring="neg_mean_squared_error", cv=5)
np.mean(mse)

# Prediction
reg_pred = regression.predict(X_test)
print(reg_pred)
from sklearn.metrics import r2_score, accuracy_score
score = r2_score(Y_test, reg_pred)
# accuracy_score = accuracy_score(Y_test, reg_pred) * 100
print(score)


