from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from ID3 import ID3DecisionTreeClassifier

data, target = datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

print(classification_report(y_test, y_hat))
print(confusion_matrix(y_test, y_hat))

id3 = ID3DecisionTreeClassifier()


id3_clf = id3.fit(X_train, y_train, attributes, None)










