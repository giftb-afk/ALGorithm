import os

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ENSEMBLE: weighted average of Random Forest (settings from RF_OG.py) and Logistic Regression
#   final probability = w * RF + (1 - w) * LogReg
# Both models are trained on the same folds, the weight w is chosen on the out-of-fold predictions


# CONFIG
# Files
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
ORIGINAL_PATH = "Churn_Modelling.csv"
SUBMISSION_PATH = "submission_Ensemble_LogReg_RF_OG.csv"

# Cross validation
N_SPLITS = 5
RANDOM_STATE = 10

# Features
OLD_AGE = 40  # age limit for AgeXInactive

# Random Forest (RF_OG.py used max_depth=10 and max_features="sqrt")
N_TREES = 470
DEPTH_VALUES = [10, 12, 14, 16]
FEATURE_VALUES = ["sqrt", 0.3, 0.5]

# Logistic Regression
LOGREG_MAX_ITER = 2000

# Blend weight search: w_RF = 0.0, 0.1, ..., 1.0
WEIGHT_STEP = 0.1

# Submission: round to 4 decimals, no prediction exactly 0 or 1
DECIMALS = 4
CLIP_MIN = 0.0001
CLIP_MAX = 0.9999


# DATA IMPORT
trainset = pd.read_csv(TRAIN_PATH)
testset = pd.read_csv(TEST_PATH)

# Original bank churn data, required: without it the RF and the blend weight would change silently
if not os.path.exists(ORIGINAL_PATH):
    raise FileNotFoundError(ORIGINAL_PATH + " not found. Put it in the same folder as this script.")

original = pd.read_csv(ORIGINAL_PATH)
original = original.rename(columns={"CustomerId": "CustomerID"})
print("Using original data:", original.shape[0], "extra rows\n")


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
    original[column + "_missing"] = 0

mean_imputer = SimpleImputer(strategy="mean")

trainset[numerical_columns] = mean_imputer.fit_transform(trainset[numerical_columns])
testset[numerical_columns] = mean_imputer.transform(testset[numerical_columns])

mode_imputer = SimpleImputer(strategy="most_frequent")

trainset[categorical_columns] = mode_imputer.fit_transform(trainset[categorical_columns])
testset[categorical_columns] = mode_imputer.transform(testset[categorical_columns])

# Original data is cleaned with the same values as the Kaggle data
original[numerical_columns] = mean_imputer.transform(original[numerical_columns])
original[categorical_columns] = mode_imputer.transform(original[categorical_columns])

datasets = [trainset, testset, original]

for df in datasets:
    df["HasCrCard"] = df["HasCrCard"].astype(float)
    df["IsActiveMember"] = df["IsActiveMember"].astype(float)

# IsSynthetic: 1 = Kaggle data, 0 = original data
trainset["IsSynthetic"] = 1
testset["IsSynthetic"] = 1
original["IsSynthetic"] = 0


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
    df["AgeXInactive"] = ((df["Age"] >= OLD_AGE) & (df["IsActiveMember"] == 0)).astype(int)

    # Feature from RF_OG.py: age of inactive members, 0 for active members
    df["AgeTimesInactive"] = df["Age"] * (1 - df["IsActiveMember"])

    # For LogReg: churn rises with age and falls again for old customers,
    # a straight line cannot show that, Age^2 and Age^3 allow a curve
    df["Age2"] = df["Age"] ** 2
    df["Age3"] = df["Age"] ** 3

# Same features as RF_OG.py plus the missing flags (CV 0.3362 -> 0.3355)
rf_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Gender_binary",
    "France",
    "Germany",
    "Spain",
    "AgeTimesInactive",
    "ManyProducts",
    "IsSynthetic",
    "Age_missing",
    "NumOfProducts_missing",
    "IsActiveMember_missing",
    "Geography_missing",
    "Balance_missing"]

# Features from testing_logReg.py plus the age curve (CV 0.3405 -> 0.3346)
# (TwoProducts and the missing flags were tested, they did not help the LogReg)
logreg_features = [
    "Age",
    "Age2",
    "Age3",
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
    "AgeXInactive"]

y = trainset["Exited"]

X_original = original[rf_features]
y_original = original["Exited"]


# MODELS
def create_rf(depth, max_feat):
    # Same model settings as RF_OG.py, but max_depth and max_features are tuned below, no calibration
    return RandomForestClassifier(
        n_estimators=N_TREES,
        # Maximum number of questions per tree, deeper = finer but more overfitting
        max_depth=depth,
        # Share of features each split may choose from
        max_features=max_feat,
        n_jobs=-1,
        random_state=RANDOM_STATE)


def create_logreg():
    # Scaling is needed for logistic regression
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=LOGREG_MAX_ITER))


# K FOLD VALIDATION
kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE)


def oof_predictions(model, features, use_original_data):
    # Trains the model on every fold and returns the out-of-fold predictions:
    # every Kaggle row gets a prediction from the model that did not see it
    predictions = np.zeros(len(trainset))

    for train_indices, validate_indices in kf.split(trainset):
        X_train = trainset[features].iloc[train_indices]
        y_train = y.iloc[train_indices]

        # Original data is only added to training, validation stays Kaggle data only
        if use_original_data:
            X_train = pd.concat([X_train, original[features]])
            y_train = pd.concat([y_train, y_original])

        model.fit(X_train, y_train)
        predictions[validate_indices] = model.predict_proba(trainset[features].iloc[validate_indices])[:, 1]

    return predictions


# TUNING: depth of the RF trees together with features per split (RF alone)
# The predictions are kept, so the best RF does not have to be trained again for the blend
rf_results = {}
rf_predictions = {}

for depth in DEPTH_VALUES:
    for max_feat in FEATURE_VALUES:
        predictions = oof_predictions(create_rf(depth, max_feat), rf_features, use_original_data=True)

        rf_predictions[(depth, max_feat)] = predictions
        rf_results[(depth, max_feat)] = log_loss(y, predictions)

        print("max_depth =", depth, " max_features =", max_feat, " Average log loss:", round(rf_results[(depth, max_feat)], 4))

best_depth, best_max_feat = min(rf_results, key=rf_results.get)

print("\nBest max_depth:", best_depth, " Best max_features:", best_max_feat,
      " Average log loss:", round(rf_results[(best_depth, best_max_feat)], 4), "\n")


# OUT-OF-FOLD PREDICTIONS for the blend
rf_oof = rf_predictions[(best_depth, best_max_feat)]

# LogReg: only Kaggle data, the original data made the LogReg worse
logreg_oof = oof_predictions(create_logreg(), logreg_features, use_original_data=False)


# WEIGHT SEARCH: w = share of RF (w = 1 is RF only, w = 0 is LogReg only)
results = {}

for w in np.round(np.arange(0, 1 + WEIGHT_STEP / 2, WEIGHT_STEP), 2):
    results[w] = log_loss(y, w * rf_oof + (1 - w) * logreg_oof)

    print("w_RF =", w, " w_LogReg =", round(1 - w, 2), " Average log loss:", round(results[w], 4))

best_w = min(results, key=results.get)

print("\nRF alone      Average log loss:", round(results[1.0], 4))
print("LogReg alone  Average log loss:", round(results[0.0], 4))
print("Best w_RF:", best_w, " Best w_LogReg:", round(1 - best_w, 2),
      " Average log loss:", round(results[best_w], 4), "\n")


# FINAL MODEL SUBMISSION
# Both models are trained on all data, then blended with the best weight
rf = create_rf(best_depth, best_max_feat).fit(pd.concat([trainset[rf_features], X_original]), pd.concat([y, y_original]))
logreg = create_logreg().fit(trainset[logreg_features], y)

rf_test = rf.predict_proba(testset[rf_features])[:, 1]
logreg_test = logreg.predict_proba(testset[logreg_features])[:, 1]

test_probabilities = best_w * rf_test + (1 - best_w) * logreg_test

# Round to 4 decimals
test_probabilities = np.round(test_probabilities, DECIMALS)

# No prediction should be exactly 0 or 1 (a wrong prediction would give log(0), which is undefined)
test_probabilities = np.clip(test_probabilities, CLIP_MIN, CLIP_MAX)

submission = pd.DataFrame({
    "CustomerID": testset["CustomerID"],
    "Exited": test_probabilities})
submission.to_csv(SUBMISSION_PATH, index=False)

print(submission.head(), "\n")
