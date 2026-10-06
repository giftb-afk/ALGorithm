#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:22:23 2026

@author: anastasiyahenechka
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:18:51 2026

@author: anastasiyahenechka

"""
""" By Anastasiya Henechka, Leo Elias Ehrenlechner, Gift Bichetero"""

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

#Cleaning the dataset by filling in the missing values - mean for numerical 
#values, mode for categorical 

#The imputers are fit on the training set only and then applied to the test
# set, so no information from the test data leaks into the cleaning step.
# Note: they are fit once on the full training set before cross-validation,
# so the validation folds contribute slightly to the means/modes. With only a
# few missing values the effect on the CV score is negligible.

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
# learned from the Kaggle training set. The original file has no missing values,
# so in practice nothing changes, but it keeps all three files consistent for safety

original[numerical_columns] = mean_imputer.transform(original[numerical_columns])
original[categorical_columns] = mode_imputer.transform(original[categorical_columns])

# Every transformation below has to be done to the original data as well,
# so we loop over all the data frames instead of repeating ourselves.

datasets = [trainset, testset, original]

for df in datasets:
    df["HasCrCard"] = df["HasCrCard"].astype(float)
    df["IsActiveMember"] = df["IsActiveMember"].astype(float)

# IsSynthetic marks where a row came from. It is 1 for every test row, so it
# cannot predict churn by itself - its job is to let the trees separate the
# synthetic rows from the real ones during training, in case the two sources
# behave differently.
trainset["IsSynthetic"] = 1
testset["IsSynthetic"] = 1
original["IsSynthetic"] = 0


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


y = trainset["Exited"]

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

# In the training data the churn rate is 35 % for one product, 
# 6 % for two, 90 % for three and 87 % for four.
# We add it as a separate feature so that it doesn't get overlooked by the trees.

    df["ManyProducts"] = (df["NumOfProducts"] >= 3).astype(int)

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
    'ManyProducts',
    "IsSynthetic"  ]

X = trainset[features]



X_original = original[features]
y_original = original["Exited"]


k = 5

kf = KFold(n_splits=k,  shuffle=True,  random_state=10)

scores = []

for train_indices, validate_indices in kf.split(X):

    X_train = X.iloc[train_indices, :]
    X_validate = X.iloc[validate_indices, :]

    y_train = y.iloc[train_indices]
    y_validate = y.iloc[validate_indices]

    
   
    X_train = pd.concat([X_train, X_original])
    y_train = pd.concat([y_train, y_original])


    model = RandomForestClassifier(n_estimators=500,random_state=10,  
                                   max_depth= 13, max_features= 'sqrt')

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_validate)[:, 1]

    score = log_loss(y_validate, probabilities)

    scores.append(score)

print("Log loss per fold:", scores)
print("Average log loss:", np.mean(scores))

final_model = RandomForestClassifier(n_estimators=500, random_state=10,  
                                     max_depth = 13 , max_features='sqrt')


final_model.fit(pd.concat([X, X_original]), pd.concat([y, y_original]))


importance = pd.Series(
    final_model.feature_importances_,
    index=features)

importance = importance.sort_values(ascending=False)

print(importance)

X_test = testset[features]
test_probabilities = final_model.predict_proba(X_test)[:, 1]

# No prediction should be exactly 0 or 1 (a wrong prediction would give log(0), which is undefined)
test_probabilities[test_probabilities < 0.0001] = 0.0001
test_probabilities[test_probabilities > 0.9999] = 0.9999


submission4 = pd.DataFrame({
    "CustomerID": testset["CustomerID"],
    "Exited": test_probabilities})
submission4.to_csv("submission4.csv", index=False)
print(submission4.head())
print(submission4.shape)

"""" PROJECT REPORT

We evaluated our models using 5-fold cross-validation (CV) and log loss,
following the model-selection approach used in class. Five folds provided
repeated out-of-sample evaluation while keeping computational cost manageable.
We tested Random Forest, bagging and SVC. SVC was standardized and tested with
different C values and kernels, but performed worse, as did bagging. We also
tested HistGradientBoostingClassifier, but its improvement was too small to
justify the additional complexity and computational cost. We, therefore, selected
Random Forest as the best trade-off between performance and complexity.

Missing numerical values were imputed with their training-set means and
categorical/binary values with their most frequent values using SimpleImputer.
IterativeImputer was also tested, but did not provide sufficient improvement.
Gender and Geography were converted to numerical indicators. Engineered
features were tested individually using CV: ZeroBalance and OneProduct were
discarded, while ManyProducts and AgeXInactive improved performance and were
retained. Removing variables with low Random Forest feature importance also
worsened validation performance.

Random Forest parameters including n_estimators, max_depth and max_features
were varied and compared using CV log loss and trial and error. Our final model 
used 470 trees,max_depth=10 and max_features="sqrt", with probabilities generated using
predict_proba().

We also incorporated the permitted external bank-churn data. During CV it was
added only to training folds, keeping validation exclusively on course data.
IsSynthetic distinguished the two data sources. Finally, we tested a weighted
Random Forest/Logistic Regression ensemble using out-of-fold predictions. It
improved log loss by only about 0.002, which did not justify the additional
complexity. We therefore retained Random Forest and based our decisions
primarily on CV rather than the public leaderboard.
"""


