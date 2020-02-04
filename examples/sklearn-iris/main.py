#!/usr/bin/env python
import sys

from orion.client import report_results
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

hyper_epsilon = sys.argv[1]
print("Epsilon is {}".format(hyper_epsilon))

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = SGDClassifier(random_state=42, loss='huber', epsilon=float(hyper_epsilon))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = balanced_accuracy_score(y_test, y_pred)

print("Accuracy is {}".format(accuracy))

report_results([dict(
    name='accuracy',
    type='objective',
    value=1 - accuracy)])
