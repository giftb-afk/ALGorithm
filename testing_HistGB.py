import os

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier


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
    "IsSynthetic"
]

y = trainset["Exited"]
X = trainset[features]
X_test = testset[features]

if use_original:
    X_original = original[features]
    y_original = original["Exited"]


# MODEL
def make_model(n_trees, n_leaves):
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=n_trees,
        max_leaf_nodes=n_leaves,
        # Off, otherwise it turns on automatically with more than 10000 rows (Kaggle + original data)
        early_stopping=False,
        random_state=10)


# K FOLD VALIDATION
k = 5

kf = KFold(
    n_splits=k,
    shuffle=True,
    random_state=10)

# TUNING STEP 2: size of each tree (max_leaf_nodes) together with number of trees (max_iter)
# Smaller trees learn less per tree, so they usually need more trees
leaf_values = [4, 8, 16, 31]
tree_values = [50, 75, 100, 150, 200, 300]

results = {}

for n_leaves in leaf_values:
    for n_trees in tree_values:
        scores = []

        for train_indices, validate_indices in kf.split(X):
            X_train = X.iloc[train_indices, :]
            X_validate = X.iloc[validate_indices, :]

            y_train = y.iloc[train_indices]
            y_validate = y.iloc[validate_indices]

            # Original data is only added to training, validation stays Kaggle data only
            if use_original:
                X_train = pd.concat([X_train, X_original])
                y_train = pd.concat([y_train, y_original])

            model = make_model(n_trees, n_leaves)
            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_validate)[:, 1]

            score = log_loss(y_validate, probabilities)

            scores.append(score)

        results[(n_trees, n_leaves)] = np.mean(scores)

        print("max_leaf_nodes =", n_leaves, " max_iter =", n_trees, " Average log loss:", round(np.mean(scores), 4))

best_n_trees, best_n_leaves = min(results, key=results.get)

print("\nBest max_leaf_nodes:", best_n_leaves, " Best max_iter:", best_n_trees,
      " Average log loss:", round(results[(best_n_trees, best_n_leaves)], 4), "\n")


# FINAL MODEL SUBMISSION
final_model = make_model(best_n_trees, best_n_leaves)

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
submission.to_csv("submission_HistGB.csv", index=False)

print(submission.shape)
print(submission.head(), "\n")
