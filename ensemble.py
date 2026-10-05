import os

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ENSEMBLE: weighted average of Random Forest and Logistic Regression probabilities
#   final probability = w * RF + (1 - w) * LogReg
# Both models are trained on the same folds, the weight w is chosen on the out-of-fold predictions


# DATA IMPORT
trainset = pd.read_csv("train.csv")
testset = pd.read_csv("test.csv")

# Original bank churn data, only used if the file is in the folder
original_path = "Churn_Modelling.csv"
use_original = os.path.exists(original_path)

if use_original:
    original = pd.read_csv(original_path)
    original = original.rename(columns={"CustomerId": "CustomerID"})
    print("Using original data:", original.shape[0], "extra rows\n")
else:
    print("Churn_Modelling.csv not found, only Kaggle data is used\n")


# DATA CLEANING (same as testing_RF_tuned.py)
numerical_columns = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "EstimatedSalary"]

# NumOfProducts is imputed with the most frequent value, so it stays a whole number
categorical_columns = [
    "Geography",
    "Gender",
    "HasCrCard",
    "IsActiveMember",
    "NumOfProducts"]

# Missing flags: 1 = value was missing before imputation (set before imputing!)
missing_flag_columns = [
    "Age",
    "NumOfProducts",
    "IsActiveMember",
    "Geography",
    "Balance"]

for column in missing_flag_columns:
    trainset[column + "_missing"] = trainset[column].isna().astype(int)
    testset[column + "_missing"] = testset[column].isna().astype(int)

    # Original data has no missing values
    if use_original:
        original[column + "_missing"] = 0

mean_imputer = SimpleImputer(strategy="mean")

trainset[numerical_columns] = mean_imputer.fit_transform(trainset[numerical_columns])
testset[numerical_columns] = mean_imputer.transform(testset[numerical_columns])

mode_imputer = SimpleImputer(strategy="most_frequent")

trainset[categorical_columns] = mode_imputer.fit_transform(trainset[categorical_columns])
testset[categorical_columns] = mode_imputer.transform(testset[categorical_columns])

datasets = [trainset, testset]

# Original data is cleaned with the same values as the Kaggle data
if use_original:
    original[numerical_columns] = mean_imputer.transform(original[numerical_columns])
    original[categorical_columns] = mode_imputer.transform(original[categorical_columns])
    datasets.append(original)

for df in datasets:
    df["HasCrCard"] = df["HasCrCard"].astype(float)
    df["IsActiveMember"] = df["IsActiveMember"].astype(float)
    df["NumOfProducts"] = df["NumOfProducts"].astype(float)

# IsSynthetic: 1 = Kaggle data, 0 = original data
trainset["IsSynthetic"] = 1
testset["IsSynthetic"] = 1

if use_original:
    original["IsSynthetic"] = 0


old_age = 40

for df in datasets:

    # Make gender binary
    df["Gender_binary"] = df["Gender"].map({"Male": 0, "Female": 1})

    # Make geography binary
    df["France"] = (df["Geography"] == "France").astype(int)
    df["Germany"] = (df["Geography"] == "Germany").astype(int)
    df["Spain"] = (df["Geography"] == "Spain").astype(int)

    # New features
    df["OneProduct"] = (df["NumOfProducts"] == 1).astype(int)
    # Customers with exactly 2 products churn the least
    df["TwoProducts"] = (df["NumOfProducts"] == 2).astype(int)
    df["ManyProducts"] = (df["NumOfProducts"] >= 3).astype(int)
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["AgeXInactive"] = ((df["Age"] >= old_age) & (df["IsActiveMember"] == 0)).astype(int)

features = [
    "Age",
    "NumOfProducts",
    "IsActiveMember",
    "France",
    "Germany",
    "Spain",
    "Gender_binary",
    "Balance",
    "CreditScore",
    "Tenure",
    "HasCrCard",
    "EstimatedSalary",
    "OneProduct",
    "TwoProducts",
    "ManyProducts",
    "ZeroBalance",
    "AgeXInactive",
    "IsSynthetic",
    "Age_missing",
    "NumOfProducts_missing",
    "IsActiveMember_missing",
    "Geography_missing",
    "Balance_missing"]

y = trainset["Exited"]
X = trainset[features]
X_test = testset[features]

if use_original:
    X_original = original[features]
    y_original = original["Exited"]
else:
    # Empty, so pd.concat below works the same without original data
    X_original = X.iloc[:0]
    y_original = y.iloc[:0]


# MODELS
def create_rf():
    # Best setting from testing_RF_tuned.py (calibrated tuning, 5x3 repeated CV)
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=0.2,
        n_jobs=-1,
        random_state=10)

    # Isotonic calibration was better than sigmoid for every combination
    return CalibratedClassifierCV(rf, method="isotonic", cv=5)


def create_logreg():
    # Scaling is needed for logistic regression
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000))


# REPEATED K FOLD VALIDATION
# 5 folds x 3 repeats, stratified: same folds as testing_RF_tuned.py
kf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=10)

# For every fold we keep the true labels and the predictions of both models,
# so different weights can be tested later without training again
fold_results = []

for train_indices, validate_indices in kf.split(X, y):
    # Original data is only added to training, validation stays Kaggle data only
    X_train = pd.concat([X.iloc[train_indices], X_original])
    y_train = pd.concat([y.iloc[train_indices], y_original])
    X_validate = X.iloc[validate_indices]

    rf = create_rf().fit(X_train, y_train)
    logreg = create_logreg().fit(X_train, y_train)

    fold_results.append((
        y.iloc[validate_indices],
        rf.predict_proba(X_validate)[:, 1],
        logreg.predict_proba(X_validate)[:, 1]))


# WEIGHT SEARCH: w = share of RF (w = 1 is RF only, w = 0 is LogReg only)
results = {}

for w in np.round(np.arange(0, 1.01, 0.05), 2):
    scores = []

    for y_validate, rf_probabilities, logreg_probabilities in fold_results:
        probabilities = w * rf_probabilities + (1 - w) * logreg_probabilities
        scores.append(log_loss(y_validate, probabilities))

    results[w] = np.mean(scores)

    print("w_RF =", w, " w_LogReg =", round(1 - w, 2), " Average log loss:", round(results[w], 4))

best_w = min(results, key=results.get)

print("\nRF alone      Average log loss:", round(results[1.0], 4))
print("LogReg alone  Average log loss:", round(results[0.0], 4))
print("Best w_RF:", best_w, " Best w_LogReg:", round(1 - best_w, 2),
      " Average log loss:", round(results[best_w], 4), "\n")


# FINAL MODEL SUBMISSION
# Both models are trained on all data, then blended with the best weight
X_full = pd.concat([X, X_original])
y_full = pd.concat([y, y_original])

rf = create_rf().fit(X_full, y_full)
logreg = create_logreg().fit(X_full, y_full)

test_probabilities = best_w * rf.predict_proba(X_test)[:, 1] + (1 - best_w) * logreg.predict_proba(X_test)[:, 1]

# Clipping probabilities to avoid log(0) errors in log loss calculation
test_probabilities = np.clip(test_probabilities, 0.0001, 0.9999)

submission = pd.DataFrame({
    "CustomerID": testset["CustomerID"],
    "Exited": test_probabilities})
submission.to_csv("submission_Ensemble.csv", index=False)

print(submission.head(), "\n")
