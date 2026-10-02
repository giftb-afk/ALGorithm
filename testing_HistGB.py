import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier


# DATA IMPORT
trainset = pd.read_csv("train.csv")
testset = pd.read_csv("test.csv")


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

for col in ["HasCrCard", "IsActiveMember"]:
    trainset[col] = trainset[col].astype(float)
    testset[col] = testset[col].astype(float)


old_age = 40

for df in [trainset, testset]:

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
    "AgeXInactive"
]

y = trainset["Exited"]
X = trainset[features]
X_test = testset[features]


# MODEL
def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        random_state=10)


# K FOLD VALIDATION
k = 5

kf = KFold(
    n_splits=k,
    shuffle=True,
    random_state=10)

scores = []

for train_indices, validate_indices in kf.split(X):
    X_train = X.iloc[train_indices, :]
    X_validate = X.iloc[validate_indices, :]

    y_train = y.iloc[train_indices]
    y_validate = y.iloc[validate_indices]

    model = make_model()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_validate)[:, 1]

    score = log_loss(y_validate, probabilities)

    scores.append(score)

print("HistGB")
print("Log loss per fold:", np.round(scores, 4))
print("Average log loss:", round(np.mean(scores), 4), "\n")


# FINAL MODEL SUBMISSION
final_model = make_model()
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
