import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# Load iris dataset
iris = load_iris()

df = sns.load_dataset('iris')

# Independent and dependent features
X = df.iloc[:, :-1]
Y = iris.target

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Model Training Post Pruning
treemodel = DecisionTreeClassifier(max_depth=2)
treemodel.fit(X_train, y_train)

from sklearn import tree
plt.figure(figsize=(15, 10))
tree.plot_tree(treemodel, filled=True)
plt.show()

y_pred = treemodel.predict(X_test)
print("Decision Tree Post Pruning")
print("y_pred: ", y_pred)

from sklearn.metrics import accuracy_score, classification_report
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

# Now Pre Pruning Decision Tree
print("Decision Tree Pre Pruning")
parameter = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_features': ['auto', 'sqrt', 'log2']
}

from sklearn.model_selection import GridSearchCV
treemodel = DecisionTreeClassifier(max_depth=2)
grid = GridSearchCV(treemodel, parameter, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameter: ", grid.best_params_)
print("Prediction: ",grid.predict(X_test))
print("Accuracy: ", accuracy_score(y_test, grid.predict(X_test)))

print("Classification Report: \n", classification_report(y_test, grid.predict(X_test)))


