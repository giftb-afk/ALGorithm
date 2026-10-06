#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:18:51 2026

@author: anastasiyahenechka

"""
""" By Anastasiya Henechka, Leo Elias Ehrenlechner, Gift Bichetero"""

# ENSEMBLE: weighted average of the Random Forest (settings from project.py)
# and a Logistic Regression:
#   final probability = w * RF + (1 - w) * LogReg
# Both models are trained on the same folds, the weight w is chosen on the
# out-of-fold predictions.

#WARNING! og_bank_train.csv (the permitted external bank-churn data) must be in
# the same folder as this script, otherwise pd.read_csv raises a FileNotFoundError.
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

trainset = pd.read_csv("bank_train.csv")
testset = pd.read_csv("bank_test.csv")


original_path = "og_bank_train.csv"

original = pd.read_csv(original_path)

#in the original the column is called CustomerId; on Kaggle it is CustomerID
original = original.rename(columns={"CustomerId": "CustomerID"})


numerical_columns = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"]

categorical_columns = [
    "Geography",
    "Gender",
    "HasCrCard",
    "IsActiveMember"]

# Missing flags: 1 = the value was missing before imputation. They have to be
# set before imputing, otherwise there is nothing missing anymore.
missing_flag_columns = [
    "Age",
    "NumOfProducts",
    "IsActiveMember",
    "Geography",
    "Balance"]

for column in missing_flag_columns:
    trainset[column + "_missing"] = trainset[column].isna().astype(int)
    testset[column + "_missing"] = testset[column].isna().astype(int)
# The original data has no missing values
    original[column + "_missing"] = 0

#Cleaning the dataset by filling in the missing values - mean for numerical
#values, mode for categorical

#The imputers are fit on the training set only and then applied to the test
# set, so no information from the test data leaks into the cleaning step.

mean_imputer = SimpleImputer(strategy="mean")

trainset[numerical_columns] = mean_imputer.fit_transform(
    trainset[numerical_columns])

testset[numerical_columns] = mean_imputer.transform(
    testset[numerical_columns])

mode_imputer = SimpleImputer(strategy="most_frequent")

trainset[categorical_columns] = mode_imputer.fit_transform(
    trainset[categorical_columns])

testset[categorical_columns] = mode_imputer.transform(
    testset[categorical_columns])

# The original data is cleaned with the same imputers, i.e. with the means
# learned from the Kaggle training set, so all three files stay consistent.

original[numerical_columns] = mean_imputer.transform(original[numerical_columns])
original[categorical_columns] = mode_imputer.transform(original[categorical_columns])

# Every transformation below has to be done to the original data as well,
# so we loop over all the data frames instead of repeating ourselves.

datasets = [trainset, testset, original]

for df in datasets:
    df["HasCrCard"] = df["HasCrCard"].astype(float)
    df["IsActiveMember"] = df["IsActiveMember"].astype(float)

# IsSynthetic marks where a row came from (1 = Kaggle data, 0 = original data).
trainset["IsSynthetic"] = 1
testset["IsSynthetic"] = 1
original["IsSynthetic"] = 0


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


y = trainset["Exited"]

old_age = 40

# Make gender binary, make geography binary, add the new features.

for df in datasets:

    df["Gender_binary"] = df["Gender"].map({
        "Male": 0,
        "Female": 1})
# This is called one hot encoding
    df["France"] = (df["Geography"] == "France").astype(int)
    df["Germany"] = (df["Geography"] == "Germany").astype(int)
    df["Spain"] = (df["Geography"] == "Spain").astype(int)

# Older, inactive customers are more likely to churn

    df["AgeXInactive"] = (df["Age"] * (1 - df["IsActiveMember"]))

# The same idea as a 0/1 indicator, used by the Logistic Regression

    df["OldInactive"] = ((df["Age"] >= old_age) &
                         (df["IsActiveMember"] == 0)).astype(int)

# In the training data the churn rate is 35 % for one product,
# 6 % for two, 90 % for three and 87 % for four.

    df["OneProduct"] = (df["NumOfProducts"].round() == 1).astype(int)
    df["ManyProducts"] = (df["NumOfProducts"] >= 3).astype(int)
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

# Same features as project.py plus the missing flags (CV 0.3362 -> 0.3355)
rf_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    'Gender_binary',
    "France",
    "Germany",
    "Spain",
    "AgeXInactive",
    'ManyProducts',
    "IsSynthetic",
    "Age_missing",
    "NumOfProducts_missing",
    "IsActiveMember_missing",
    "Geography_missing",
    "Balance_missing"  ]

# Same features as testing_logReg.py
# (TwoProducts and the missing flags were tested, they did not help the LogReg)
logreg_features = [
    "Age",
    "NumOfProducts",
    "IsActiveMember",
    "France",
    "Germany",
    "Spain",
    'Gender_binary',
    "Balance",
    "CreditScore",
    "Tenure",
    "HasCrCard",
    "EstimatedSalary",
    "OneProduct",
    'ManyProducts',
    "ZeroBalance",
    "OldInactive"  ]

X_rf = trainset[rf_features]
X_logreg = trainset[logreg_features]



X_original = original[rf_features]
y_original = original["Exited"]


k = 5

kf = KFold(n_splits=k,  shuffle=True,  random_state=10)

# For every fold we keep the true labels and the predictions of both models,
# so different weights can be tested later without training again.
fold_results = []

for train_indices, validate_indices in kf.split(X_rf):

    y_train = y.iloc[train_indices]
    y_validate = y.iloc[validate_indices]

# RF: the original data is only added to the training folds,
# validation stays Kaggle data only

    X_train_rf = X_rf.iloc[train_indices, :]
    X_validate_rf = X_rf.iloc[validate_indices, :]

    X_train_rf = pd.concat([X_train_rf, X_original])
    y_train_rf = pd.concat([y_train, y_original])


    rf_model = RandomForestClassifier(n_estimators=470,random_state=10,
                                      max_depth= 10, max_features= 'sqrt')

    rf_model.fit(X_train_rf, y_train_rf)

    rf_probabilities = rf_model.predict_proba(X_validate_rf)[:, 1]

# LogReg: only Kaggle data, the original data made the LogReg worse.
# Scaling is needed for logistic regression.

    X_train_logreg = X_logreg.iloc[train_indices, :]
    X_validate_logreg = X_logreg.iloc[validate_indices, :]

    logreg_model = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000))

    logreg_model.fit(X_train_logreg, y_train)

    logreg_probabilities = logreg_model.predict_proba(X_validate_logreg)[:, 1]

    fold_results.append((y_validate, rf_probabilities, logreg_probabilities))

# Weight search: w = share of the RF (w = 1 is RF only, w = 0 is LogReg only)
results = {}

for w in np.round(np.arange(0, 1.01, 0.1), 1):

    scores = []

    for y_validate, rf_probabilities, logreg_probabilities in fold_results:

        probabilities = w * rf_probabilities + (1 - w) * logreg_probabilities

        score = log_loss(y_validate, probabilities)

        scores.append(score)

    results[w] = np.mean(scores)

    print("w_RF =", w, " w_LogReg =", round(1 - w, 1),
          " Average log loss:", round(results[w], 4))

best_w = min(results, key=results.get)

print("RF alone      Average log loss:", round(results[1.0], 4))
print("LogReg alone  Average log loss:", round(results[0.0], 4))
print("Best w_RF:", best_w, " Best w_LogReg:", round(1 - best_w, 1),
      " Average log loss:", round(results[best_w], 4))

# Both final models are trained on all the data, then blended with the best weight

final_rf = RandomForestClassifier(n_estimators=470, random_state=10,
                                  max_depth = 10 , max_features='sqrt')


final_rf.fit(pd.concat([X_rf, X_original]), pd.concat([y, y_original]))

final_logreg = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000))

final_logreg.fit(X_logreg, y)


importance = pd.Series(
    final_rf.feature_importances_,
    index=rf_features)

importance = importance.sort_values(ascending=False)

print(importance)

rf_test = final_rf.predict_proba(testset[rf_features])[:, 1]
logreg_test = final_logreg.predict_proba(testset[logreg_features])[:, 1]

test_probabilities = best_w * rf_test + (1 - best_w) * logreg_test

# No prediction should be exactly 0 or 1 (a wrong prediction would give log(0), which is undefined)
test_probabilities[test_probabilities < 0.0001] = 0.0001
test_probabilities[test_probabilities > 0.9999] = 0.9999


submission_ensemble = pd.DataFrame({
    "CustomerID": testset["CustomerID"],
    "Exited": test_probabilities})
submission_ensemble.to_csv("submission_Ensemble_LogReg_RF_OG.csv", index=False)
print(submission_ensemble.head())
print(submission_ensemble.shape)
