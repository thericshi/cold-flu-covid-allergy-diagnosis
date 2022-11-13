import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv("symptom_data.csv")

train = train.drop(range(0, 15357))
print(train)
train = train.drop(range(18429, 43429))

X = train.drop("TYPE", axis=1)
y = train["TYPE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64)

model = LogisticRegression(multi_class="multinomial")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


df = pd.read_csv("test.csv")
print(df)

print(model.predict(df))

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))
# plt.show()
