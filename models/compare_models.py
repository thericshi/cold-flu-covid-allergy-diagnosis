import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, neural_network, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Load the data
train = pd.read_csv("under-sampled.csv")

# Split the data into a training and a testing set
X = train.drop("TYPE", axis=1)
y = train["TYPE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

# Create an SVM model and train it on the training data
svm_model = svm.SVC(kernel="linear")
svm_model.fit(X_train, y_train)

# Make predictions on the test set and evaluate the performance
svm_predictions = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, svm_predictions))
print(confusion_matrix(y_test, svm_predictions))

# Create a k-NN model and train it on the training data
knn_model = neighbors.KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Make predictions on the test set and evaluate the performance
knn_predictions = knn_model.predict(X_test)
print("k-NN Classification Report:")
print(classification_report(y_test, knn_predictions))
print(confusion_matrix(y_test, knn_predictions))

# Create a neural network model and train it on the training data
nn_model = neural_network.MLPClassifier(max_iter=1000)
nn_model.fit(X_train, y_train)

# Make predictions on the test set and evaluate the performance
nn_predictions = nn_model.predict(X_test)
print("Neural Network Classification Report:")
print(classification_report(y_test, nn_predictions))
print(confusion_matrix(y_test, nn_predictions))

# Create a logistic regression model and train it on the training data
lr_model = linear_model.LogisticRegression(solver="sag", max_iter=1000, multi_class="multinomial")
lr_model.fit(X_train, y_train)

# Make predictions on the test set
# Make predictions on the test set and evaluate the performance
lr_predictions = lr_model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))
print(confusion_matrix(y_test, lr_predictions))

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print(confusion_matrix(y_test, rf_predictions))

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))
print(confusion_matrix(y_test, dt_predictions))

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))
print(confusion_matrix(y_test, nb_predictions))

# Evaluate the models using cross-validation
svm_scores = cross_val_score(svm_model, X, y, cv=5)
knn_scores = cross_val_score(knn_model, X, y, cv=5)
mlp_scores = cross_val_score(nn_model, X, y, cv=5)
lr_scores = cross_val_score(lr_model, X, y, cv=5)
rf_scores = cross_val_score(rf_model, X, y, cv=5)
dt_scores = cross_val_score(dt_model, X, y, cv=5)
nb_scores = cross_val_score(nb_model, X, y, cv=5)

# Print the cross-validated accuracy for the models
print("SVM Cross-Validation Score:", svm_scores.mean())
print("KNN Cross-Validation Score:", knn_scores.mean())
print("MLP Cross-Validation Score:", mlp_scores.mean())
print("LR Cross-Validation Score:", lr_scores.mean())
print("RF Cross-Validation Score:", rf_scores.mean())
print("DT Cross-Validation Score:", dt_scores.mean())
print("NB Cross-Validation Score:", nb_scores.mean())
