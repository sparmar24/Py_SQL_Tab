# changes
"""dropping the 4 features (L 15)"""
"""testing data """

import pandas as pd
import numpy as np

data_preprocessed = pd.read_csv("Absenteeism_preprocessed.csv")
# data_preprocessed["Absenteeism Time in Hours"].median()

targets = np.where(
    data_preprocessed["Absenteeism Time in Hours"]
    > data_preprocessed["Absenteeism Time in Hours"].median(),
    1,
    0,
)
data_preprocessed["Excessive Absenteeism"] = targets
# print(data_preprocessed)

# total targets = 1 / total targets
targets.sum() / targets.shape[0]
# only 45% of targets are 1 and others are 0
data_with_targets = data_preprocessed.drop(
    [
        "Absenteeism Time in Hours",
        "Day of the week",
        "Daily Work Load Average",
        "Distance to Work",
    ],
    axis=1,
)

# print(data_with_targets is data_preprocessed)
# print( data_with_targets.shape)

# print(data_with_targets.iloc[:, :-1])
unscaled_input = data_with_targets.iloc[:, :-1]
# X = unscaled_input

# print(unscaled_input.columns.values)

# import the libraries needed to create the Custom Scaler
# note that all of them are a part of the sklearn package
# moreover, one of them is actually the StandardScaler module,
# so you can imagine that the Custom Scaler is build on it
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

absenteeism_scaler = StandardScaler()
# create the Custom Scaler class


class CustomScaler(BaseEstimator, TransformerMixin):

    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do

    def __init__(self, columns):  # , copy=True, with_mean=True, with_std=True):
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler()  # copy, with_mean, with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    # the fit method, which, again based on StandardScale

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        # record the initial order of the columns
        init_col_order = X.columns

        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(
            self.scaler.transform(X[self.columns]), columns=self.columns
        )

        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]

        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# print(unscaled_input.columns.values)
# columns_to_scale = ["Month Value", "Day of the week", "Transportation expense", "distance to work","age", daily work load average, body mass index, children, pets]
columns_to_omit = ["reason_1", "reason_2", "reason_3", "reason_4", "Education"]
columns_to_scale = [
    x for x in unscaled_input.columns.values if x not in columns_to_omit
]
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_input)
scaled_inputs = absenteeism_scaler.transform(unscaled_input)
# print(scaled_inputs)
# print(scaled_inputs.shape)


""" Split data"""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    scaled_inputs, targets, test_size=0.2, random_state=20
)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

""" Logistic regression"""
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

reg = LogisticRegression()
reg.fit(x_train, y_train)
reg.score(x_train, y_train)


""" accuracy from the model"""
print(reg.score(x_train, y_train))
model_outputs = reg.predict(x_train)
# print(model_outputs)

"""Manual checking of accuracy"""
# print(model_outputs == y_train)
print(np.sum((model_outputs == y_train)) / model_outputs.shape[0])

# print(reg.intercept_)
# print(reg.coef_)

# print(unscaled_input.columns.values)
feature_name = unscaled_input.columns.values
summary_table = pd.DataFrame(columns=["feature_name"], data=feature_name)
summary_table["Coefficient"] = np.transpose(reg.coef_)

summary_table.index = summary_table.index + 1
summary_table.loc[0] = ["Intercept", reg.intercept_[0]]
summary_table = summary_table.sort_index()
# print(summary_table)

"""interpreting the coefficients"""
summary_table["Odds_ratio"] = np.exp(summary_table.Coefficient)
summary_table = summary_table.sort_values("Odds_ratio", ascending=False)
print(summary_table)
## Higher the coefficient value more important the feature is


""" testing Model test-data"""
predicted_proba = reg.predict_proba(x_test)
print("test-data accuracy score", reg.score(x_test, y_test))
print(predicted_proba)
# first and second columns provide prob of being 0 and 1 resp
# -> absenteesim probability

print(predicted_proba.shape)
print(predicted_proba[:, 1])


""" Save the model"""
import pickle

with open("model", "wb") as file:
    pickle.dump(reg, file)

with open("scaler", "wb") as file:
    pickle.dump(absenteeism_scaler, file)
