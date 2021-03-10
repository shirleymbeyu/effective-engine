# -*- coding: utf-8 -*-
"""Modelling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11XpwbHlaIRXRH_lbuejyqki5cVprypOf
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("credit_data (1).csv")
df

df.info()

"""## DATA PREPARATION

Our target varaible is default status and in order to work with it, we have to binarize it as: 0:N, 1:Y
"""

# LabelBinarizer converts the string categorical variable to binary 
from sklearn.preprocessing import LabelBinarizer
lb= LabelBinarizer()
df["Default_Status"]= lb.fit_transform(df["Default_Status"])

# plotting risk distribution to understand whether there are more records 
# with more categories than the other.
sns.countplot('Default_Status', data = df);

"""Binning on the numeric variales: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term"""

df['ApplicantIncome'] = pd.qcut(df.ApplicantIncome, q = 6)

interval = (0.0 , 10000, 20000, 30000, 41667)
df['CoapplicantIncome'] = pd.cut(df.CoapplicantIncome, interval)

interval = (0, 140, 280, 320, 460, 700)
df['LoanAmount'] = pd.cut(df.LoanAmount, interval)

interval = (0 ,96, 192, 288, 384, 480)
df["Loan_Amount_Term"] = pd.cut(df.Loan_Amount_Term, interval)

df.head()

"""Tranforming our features into dummy variables by one-hot encode, hence making them robust for our linear regresssion model.
we'll set the keyword drop_first to true so that one of the unique variables is deleted.
"""

#GENDER
df = df.merge(pd.get_dummies(df.Gender, drop_first= True, prefix='sex'), left_index=True, right_index=True)

#Married
df = df.merge(pd.get_dummies(df.Married, drop_first= True, prefix='Married'), left_index=True, right_index=True)

#Dependents
df = df.merge(pd.get_dummies(df.Dependents, drop_first= True, prefix='Dependents'), left_index=True, right_index=True)

#Education
df = df.merge(pd.get_dummies(df.Education, drop_first= True, prefix='Education'), left_index=True, right_index=True)

#Self_Employed
df = df.merge(pd.get_dummies(df.Self_Employed, drop_first= True, prefix='Self_Employed'), left_index=True, right_index=True)

#Credit_History
df = df.merge(pd.get_dummies(df.Credit_History, drop_first= True, prefix='Credit_History'), left_index=True, right_index=True)

#Property_Area
df = df.merge(pd.get_dummies(df.Property_Area, drop_first= True, prefix='Property_Area'), left_index=True, right_index=True)

#ApplicantIncome
df = df.merge(pd.get_dummies(df.ApplicantIncome, drop_first= True, prefix='ApplicantIncome'), left_index=True, right_index=True)

#CoapplicantIncome
df = df.merge(pd.get_dummies(df.CoapplicantIncome, drop_first= True, prefix='CoapplicantIncome'), left_index=True, right_index=True)

#LoanAmount
df = df.merge(pd.get_dummies(df.LoanAmount, drop_first= True, prefix='LoanAmount'), left_index=True, right_index=True)

#Loan_Amount_Term
df = df.merge(pd.get_dummies(df.Loan_Amount_Term, drop_first= True, prefix='Loan_Amount_Term'), left_index=True, right_index=True)

"""Preview our created data frame"""

df.head()

#we exclude the other columns since we have new ones
del df["Gender"]
del df["Married"]
del df["Dependents"]
del df["Education"]
del df["Self_Employed"]
del df["ApplicantIncome"]
del df["CoapplicantIncome"]
del df["LoanAmount"]
del df["Loan_Amount_Term"]
del df["Credit_History"]
del df["Property_Area"]

df.head()

"""##Model preparation"""

# dividing our dataset into features (X) and target (y)
X = df.drop(columns = ['Default_Status']).values
y = df['Default_Status'].values

print(X.shape)
print(y.shape)

# splitting our dataset into 80-20 train-test sets
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

"""Because we had earlier seen that we had an imbalanced dataset, we will create a balanced dataset by trying to resample our dataset using SMOTE (Synthetic minority Oversampling Technique). This technique works randomly picking a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors."""

# creating a balanced dataset
from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)

# we check the amount of records in each category
np.bincount(y_train)

"""#MODELLING

###(a) Logistic Regression
"""

# model creation
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression()

# training our model
logistic_classifier.fit(X_train, y_train)

# making predictions
y_pred_logistic = logistic_classifier.predict(X_test)

"""#MODEL EVALUATION"""

# model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_pred_logistic, y_test))
print(confusion_matrix(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

"""The acccuracy of our model is 0.71

From our confusion matrix, 3 records with class 0(not defaulting) were predicted correctly while 0 were predicted incorrectly. 84 of class 1(defaulting) were predicted correctly while 36 were predicted incorrectly.

We have a recall of 0.54 (macro avg); ability to predict positive when it was actually positive.



"""

# Exploring another metric below 
# ---
# plotting roc curve (receiving operating characteristic curve)
from sklearn.metrics import roc_curve, roc_auc_score

# Create true and false positive rates
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred_logistic)

# Plot ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

"""The above shows the true positive rate(recall) and false positive rate for every probability threshold of a binary classifier.
The higher the blue line the better the model at distingushing beween the positive and negative classes.
"""

# roc_auc_score
roc_auc_score(y_test, y_pred_logistic)

"""The model was fair."""