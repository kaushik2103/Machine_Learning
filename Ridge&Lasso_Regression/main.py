# Linear Regression
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np

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

# Ridge Regression Model
ridge_regressor = Ridge()
parameters = {
    'alpha': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
}
ridgecv = GridSearchCV(ridge_regressor, parameters, scoring='neg_mean_squared_error',cv=5)
ridgecv.fit(X_train, Y_train)
print(ridgecv.best_params_)
print(ridgecv.best_score_)

ridge_pred = ridgecv.predict(X_test)
print(ridge_pred)
print(ridgecv.score(X_test, Y_test))


# Lasso Regression
lasso = Lasso()
parameters = {
    'alpha': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
}
lassocv = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error',cv=5)
lassocv.fit(X_train, Y_train)
print(lassocv.best_params_)
print(lassocv.best_score_)

lasso_pred = lassocv.predict(X_test)
print(lasso_pred)

# r2 score
# from sklearn.metrics import r2_score, accuracy_score
# score = r2_score(Y_test, reg_pred)
# # accuracy_score = accuracy_score(Y_test, reg_pred) * 100
# print(score)


