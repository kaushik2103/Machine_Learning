import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

california_df = fetch_california_housing()

# Independent Variables and Dependent Variables
X = pd.DataFrame(california_df.data, columns=california_df.feature_names)
Y = california_df.target

# Spliting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X_train, Y_train)

# Predicting
Y_pred = regressor.predict(X_test)
print("Predicted values: ", Y_pred)

# Evaluating
from sklearn.metrics import r2_score

score = r2_score(Y_test, Y_pred)
print("R2 Score: ", score)

# Hyperparameter Tuning
print("Hyperparameter Tuning: ")
parameters = {
    'criterion': ['friedman_mse', 'poisson', 'squared_error', 'absolute_error'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 6, 8, 10],
}

regressor_2 = DecisionTreeRegressor()

# Grid Search CV
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator=regressor_2, param_grid=parameters, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, Y_train)

print("Best Parameters: ", grid.best_params_)
Y_pred_2 = grid.predict(X_test)
print("Predicted values: ", Y_pred_2)
score_2 = r2_score(Y_test, Y_pred_2)
print("R2 Score: ", score_2)
