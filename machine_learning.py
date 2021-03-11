# -*- coding: utf-8 -*-
"""Machine_Learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C9QeZcHaN2-VlQUqGtez8wgaOLhKR9Rk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""## Data preparation"""

df = pd.read_csv('credit_train.csv')
df.head()



x = df.drop(columns = ['Default_Status']).values
y = df['Default_Status'].values

print(x.shape)
print(y.shape)

#splitting our data into 80-20 train_test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

#creating a balanced data set using SMOTE
from imblearn.over_sampling import SMOTE
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)

np.bincount(y_train)

"""#Modelling

## Decision Trees
"""

#importing the ML algorithm
from sklearn.tree import DecisionTreeClassifier

#Instatiating
decision_classifier = DecisionTreeClassifier()

#training the model
decision_classifier.fit(x_train, y_train)

#making the prediction
decision_y_prediction = decision_classifier.predict(x_test)

# evaluation metrics
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
print('Decision Tree')
print(accuracy_score(decision_y_prediction, y_test))
print(confusion_matrix(decision_y_prediction, y_test))
print(classification_report(decision_y_prediction, y_test))
print(roc_auc_score(decision_y_prediction, y_test))

"""## Random Forest"""

#importing the ML algo
from sklearn.ensemble import RandomForestClassifier

#instatiating
random_forest_classifier = RandomForestClassifier()

#Training the model
random_forest_classifier.fit(x_train, y_train)

#make prediction
random_forest_y_pred = random_forest_classifier.predict(x_test)

# evaluation metrics
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
print('Random Forest')
print(accuracy_score(random_forest_y_pred, y_test))
print(confusion_matrix(random_forest_y_pred, y_test))
print(classification_report(random_forest_y_pred, y_test))
print(roc_auc_score(random_forest_y_pred, y_test))

"""## Gradient Boosting"""

#importing the ML algo
from sklearn.ensemble import GradientBoostingClassifier

#instatiating
gbm_classifier = GradientBoostingClassifier()

#training the model
gbm_classifier.fit(x_train, y_train)

#making prediction
gbm_y_pred = gbm_classifier.predict(x_test)

# evaluation metrics
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
print('Random Forest')
print(accuracy_score(gbm_y_pred, y_test))
print(confusion_matrix(gbm_y_pred, y_test))
print(classification_report(gbm_y_pred, y_test))
print(roc_auc_score(gbm_y_pred, y_test))