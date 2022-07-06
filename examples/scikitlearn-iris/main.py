import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from orion.client import report_objective

# Make the execution reproducible
np.random.seed(1)

# Parsing the value for the hyper-parameter 'epsilon' given as a command line argument.
hyper_epsilon = sys.argv[1]
print(f"Epsilon is {hyper_epsilon}")

# Loading the iris dataset and splitting it into training and testing set.
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training the model with the training set with the specified 'epsilon' to control the huber loss.
clf = SGDClassifier(loss="huber", epsilon=float(hyper_epsilon))
clf.fit(X_train, y_train)

# Evaluating the accuracy using the testing set.
y_pred = clf.predict(X_test)
accuracy = balanced_accuracy_score(y_test, y_pred)

# Reporting the results
print(f"Accuracy is {accuracy}")

report_objective(1 - accuracy)
