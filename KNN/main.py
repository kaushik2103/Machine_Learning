import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

datasets = sns.load_dataset('iris')

X = datasets.iloc[:, :-1]
Y = datasets.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
y_pred = knn_model.predict(X_test)
print("Prediction: ", y_pred)
print("Accuracy: ", accuracy_score(Y_test, y_pred))