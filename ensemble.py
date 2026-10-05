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


models = {
    "RF": create_rf,
    "LogReg": create_logreg}


# REPEATED K FOLD VALIDATION
# 5 folds x 3 repeats, stratified: same folds as testing_RF_tuned.py
k = 5
repeats = 3

kf = RepeatedStratifiedKFold(
    n_splits=k,
    n_repeats=repeats,
    random_state=10)


def get_fold_data(train_indices, validate_indices):
    X_train = X.iloc[train_indices, :]
    X_validate = X.iloc[validate_indices, :]

    y_train = y.iloc[train_indices]
    y_validate = y.iloc[validate_indices]

    # Original data is only added to training, validation stays Kaggle data only
    if use_original:
        X_train = pd.concat([X_train, X_original])
        y_train = pd.concat([y_train, y_original])

    return X_train, X_validate, y_train, y_validate


# OUT-OF-FOLD PREDICTIONS
# For every fold we keep the true labels and the predictions of both models,
# so different weights can be tested later without training again
fold_results = []

for fold, (train_indices, validate_indices) in enumerate(kf.split(X, y), start=1):
    X_train, X_validate, y_train, y_validate = get_fold_data(train_indices, validate_indices)

    fold_predictions = {"y": y_validate.values}

    for name, create_model in models.items():
        model = create_model()
        model.fit(X_train, y_train)

        fold_predictions[name] = model.predict_proba(X_validate)[:, 1]

    fold_results.append(fold_predictions)

    print("Fold", fold, "of", k * repeats, "done", flush=True)


def blend(rf_probabilities, logreg_probabilities, w):
    return w * rf_probabilities + (1 - w) * logreg_probabilities


def average_log_loss(w):
    scores = []

    for fold_predictions in fold_results:
        probabilities = blend(fold_predictions["RF"], fold_predictions["LogReg"], w)
        scores.append(log_loss(fold_predictions["y"], probabilities))

    return np.mean(scores)


# SINGLE MODELS (w = 1 is RF only, w = 0 is LogReg only)
print("\nRF alone      Average log loss:", round(average_log_loss(1.0), 4))
print("LogReg alone  Average log loss:", round(average_log_loss(0.0), 4), "\n")


# WEIGHT SEARCH: share of RF in the ensemble
weight_values = np.round(np.arange(0, 1.01, 0.05), 2)

results = {}

for w in weight_values:
    results[w] = average_log_loss(w)

    print("w_RF =", w, " w_LogReg =", round(1 - w, 2), " Average log loss:", round(results[w], 4))

best_w = min(results, key=results.get)

print("\nBest w_RF:", best_w, " Best w_LogReg:", round(1 - best_w, 2),
      " Average log loss:", round(results[best_w], 4))
print("Improvement over RF alone:", round(average_log_loss(1.0) - results[best_w], 4), "\n")


# FINAL MODEL SUBMISSION
# Both models are trained on all data, then blended with the best weight
if use_original:
    X_full = pd.concat([X, X_original])
    y_full = pd.concat([y, y_original])
else:
    X_full = X
    y_full = y

test_predictions = {}

for name, create_model in models.items():
    final_model = create_model()
    final_model.fit(X_full, y_full)

    test_predictions[name] = final_model.predict_proba(X_test)[:, 1]

test_probabilities = blend(test_predictions["RF"], test_predictions["LogReg"], best_w)

# Round to 4 decimals
test_probabilities = np.round(test_probabilities, 4)

# No prediction should be exactly 0 or 1 (a wrong prediction would give log(0), which is undefined)
test_probabilities[test_probabilities < 0.0001] = 0.0001
test_probabilities[test_probabilities > 0.9999] = 0.9999

submission = pd.DataFrame({
    "CustomerID": testset["CustomerID"],
    "Exited": test_probabilities})
submission.to_csv("submission_Ensemble.csv", index=False)

print(submission.head(), "\n")
