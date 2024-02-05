import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
import pandas as pd

df = sns.load_dataset('iris')
df = df[df.species != 'setosa']
df['species'] = df['species'].map({'versicolor': 0, 'virginica': 1})

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, Y_train)
y_pred = naive_bayes.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Report:", classification_report(Y_test, y_pred))
