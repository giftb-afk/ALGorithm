import os

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


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


# DATA CLEANING
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
    df["OneProduct"] = (df["NumOfProducts"].round() == 1).astype(int)
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
    "ManyProducts",
    "ZeroBalance",
    "AgeXInactive",
    "IsSynthetic"]

y = trainset["Exited"]
X = trainset[features]
X_test = testset[features]

if use_original:
    X_original = original[features]
    y_original = original["Exited"]


# MODEL
def create_model(min_leaf, max_feat, calibrate):
    rf = RandomForestClassifier(
        n_estimators=500,
        # Minimum rows per leaf: bigger value = smoother probabilities, less overfitting
        min_samples_leaf=min_leaf,
        # Share of features tried at each split
        max_features=max_feat,
        n_jobs=-1,
        random_state=10)

    # Isotonic calibration: corrects the RF probabilities with an internal 5-fold split
    if calibrate:
        return CalibratedClassifierCV(rf, method="isotonic", cv=5)
    return rf


# K FOLD VALIDATION
k = 5

kf = KFold(
    n_splits=k,
    shuffle=True,
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


# TUNING: leaf size (min_samples_leaf) together with features per split (max_features)
# Grid without calibration first, because it is faster
leaf_values = [5, 10, 15, 25]
feature_values = [0.2, 0.3, 0.5, "sqrt"]

results = {}

for min_leaf in leaf_values:
    for max_feat in feature_values:
        scores = []

        for train_indices, validate_indices in kf.split(X):
            X_train, X_validate, y_train, y_validate = get_fold_data(train_indices, validate_indices)

            model = create_model(min_leaf, max_feat, calibrate=False)
            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_validate)[:, 1]

            score = log_loss(y_validate, probabilities)

            scores.append(score)

        results[(min_leaf, max_feat)] = np.mean(scores)

        print("min_samples_leaf =", min_leaf, " max_features =", max_feat, " Average log loss:", round(np.mean(scores), 4))

best_min_leaf, best_max_feat = min(results, key=results.get)

print("\nBest min_samples_leaf:", best_min_leaf, " Best max_features:", best_max_feat,
      " Average log loss:", round(results[(best_min_leaf, best_max_feat)], 4), "\n")


# CALIBRATED RF: best combination with isotonic calibration (same folds)
scores = []

for train_indices, validate_indices in kf.split(X):
    X_train, X_validate, y_train, y_validate = get_fold_data(train_indices, validate_indices)

    model = create_model(best_min_leaf, best_max_feat, calibrate=True)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_validate)[:, 1]

    score = log_loss(y_validate, probabilities)

    scores.append(score)

print("RF calibrated  Average log loss:", round(np.mean(scores), 4), "\n")


# FINAL MODEL SUBMISSION
final_model = create_model(best_min_leaf, best_max_feat, calibrate=True)

# In case we don`t have the Churn_Modelling.csv file, we only use the Kaggle data for training
if use_original:
    final_model.fit(pd.concat([X, X_original]), pd.concat([y, y_original]))
else:
    final_model.fit(X, y)

test_probabilities = final_model.predict_proba(X_test)[:, 1]

# Round to 4 decimals
test_probabilities = np.round(test_probabilities, 4)

# No prediction should be exactly 0 or 1 (a wrong prediction would give log(0), which is undefined)
test_probabilities[test_probabilities < 0.0001] = 0.0001
test_probabilities[test_probabilities > 0.9999] = 0.9999

submission = pd.DataFrame({
    "CustomerID": testset["CustomerID"],
    "Exited": test_probabilities})
submission.to_csv("submission_RF.csv", index=False)

print(submission.head(), "\n")
