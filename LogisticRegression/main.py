import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

df = sns.load_dataset('iris')
df = df[df.species != 'setosa']
df['species'] = df['species'].map({'versicolor': 0, 'virginica': 1})

# Spliting the data in independent and dependent variables

X = df.iloc[:, :-1]  # independent variables
Y = df.iloc[:, -1]  # dependent variables

# Splitting the data in train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# Logistic Regression
classifier = LogisticRegression()

parameter = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'max_iter': [100, 200, 300, 400, 500]
}

classifier_regressor = GridSearchCV(classifier, parameter, scoring='accuracy', cv=5)
classifier_regressor.fit(X_train, Y_train)

print("Best Parameter: ", classifier_regressor.best_params_)
print("Best Accuracy: ", classifier_regressor.best_score_)

# Preidiction on test data
y_pred = classifier_regressor.predict(X_test)

# Accuracy on test data
score = accuracy_score(Y_test, y_pred)
print("Accuracy: ", score)

# Classification report
report = classification_report(Y_test, y_pred)
print("Classification Report: ", report)

# EDA of the data
sns.pairplot(df, hue="species")
