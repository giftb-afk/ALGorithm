import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

trainset = pd.read_csv("bank_train.csv")
testset = pd.read_csv("bank_test.csv")
numerical_columns = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"
]

categorical_columns = [
    "Geography",
    "Gender",
    "HasCrCard",
    "IsActiveMember"
]

mean_imputer = SimpleImputer(strategy="mean")

trainset[numerical_columns] = mean_imputer.fit_transform(
    trainset[numerical_columns]
)

testset[numerical_columns] = mean_imputer.transform(
    testset[numerical_columns]
)

mode_imputer = SimpleImputer(strategy="most_frequent")

trainset[categorical_columns] = mode_imputer.fit_transform(
    trainset[categorical_columns]
)

testset[categorical_columns] = mode_imputer.transform(
    testset[categorical_columns]
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


y = trainset["Exited"]

#Make gender binary 

trainset["Gender_binary"] = trainset["Gender"].map({
    "Male": 0,
    "Female": 1
})

testset["Gender_binary"] = testset["Gender"].map({
    "Male": 0,
    "Female": 1
})

# Make geography binary 

trainset["France"] = (trainset["Geography"] == "France").astype(int)
trainset["Germany"] = (trainset["Geography"] == "Germany").astype(int)
trainset["Spain"] = (trainset["Geography"] == "Spain").astype(int)

testset["France"] = (testset["Geography"] == "France").astype(int)
testset["Germany"] = (testset["Geography"] == "Germany").astype(int)
testset["Spain"] = (testset["Geography"] == "Spain").astype(int)

#Introducing new features

trainset["AgeXInactive"] = (trainset["Age"] * (1 - trainset["IsActiveMember"]))
testset["AgeXInactive"] = (testset["Age"] * (1 - testset["IsActiveMember"]))


trainset["ManyProducts"] = (trainset["NumOfProducts"] >= 3).astype(int)
testset["ManyProducts"] = (testset["NumOfProducts"] >= 3).astype(int)

features = [
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
    'ManyProducts'
]

X = trainset[features]

k = 5

kf = KFold(
    n_splits=k,
    shuffle=True,
    random_state=10
)

scores = []

for train_indices, validate_indices in kf.split(X):

    X_train = X.iloc[train_indices, :]
    X_validate = X.iloc[validate_indices, :]

    y_train = y.iloc[train_indices]
    y_validate = y.iloc[validate_indices]

    model = RandomForestClassifier(
        n_estimators=470,
        random_state=10,  max_depth= 10, max_features="sqrt"
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_validate)[:, 1]

    score = log_loss(y_validate, probabilities)

    scores.append(score)
    
print("Log loss per fold:", scores)
print("Average log loss:", np.mean(scores)) 
    
final_model = RandomForestClassifier(
    n_estimators=470,
random_state=10,  max_depth = 10 , max_features="sqrt")

final_model.fit(X, y)
importance = pd.Series(
    final_model.feature_importances_,
    index=features
)

importance = importance.sort_values(ascending=False)

print(importance)

X_test = testset[features]
test_probabilities = final_model.predict_proba(X_test)[:, 1]
submission3 = pd.DataFrame({
    "CustomerID": testset["CustomerID"],
    "Exited": test_probabilities})
submission3.to_csv("submission3.csv", index=False)
print(submission3.head())
print(submission3.shape)


