import numpy as np
import pandas as pd
import os
import xgboost
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import OneClassSVM
from sklearn.decomposition import KernelPCA, PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_validate, GridSearchCV, KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
ROOT_PATH = "/home/jupyter/Task 1/"

# Data import
x_train = pd.read_csv(ROOT_PATH + 'X_train.csv')
y_train= pd.read_csv(ROOT_PATH + 'y_train.csv')["y"]
x_test = pd.read_csv(ROOT_PATH + 'X_test.csv')

# Impute values
imp = SimpleImputer(missing_values=np.nan, strategy='median').fit(x_train)
columns = x_train.columns.tolist()
x_train = pd.DataFrame(imp.transform(x_train), columns=columns)

# Remove zero variance features
temp = x_train.nunique(axis=0)
temp = temp[temp == 1]  # constant features
const_features = temp.index.values.tolist()
print(const_features)

x_train = x_train.drop(const_features, axis=1)



def outlier_via_ID_matrix(x_train, y_train):
    """
    This is not very stable --> if k == 1.5 or k == 1 --> the score decreases!
    """
    data_x = np.array(x_train)
    data_y = np.array(y_train)

    hat_matrix = np.matmul(np.matmul(data_x, np.linalg.inv(np.matmul(data_x.transpose(), data_x))),data_x.transpose()) 
    # Hat Matrix should be an identity --> All the indexes where the value is quite small are likely outliers
    p_ii = (list(hat_matrix.diagonal()))
    # sns.displot(p_ii)
    # plt.show()

    # Consider as outlier all that are 3 sigma away from mean
    avg = hat_matrix.diagonal().mean()
    dev = hat_matrix.diagonal().std()
    # print(avg - 1.35*dev)  # 3 sigma cca 99.7%
    k = 1.35
    indexes = [i for i in range(len(p_ii)) if p_ii[i] > avg - k*dev]
    outliers = [1 if p_ii[i] > avg - k*dev else -1 for i in range(len(p_ii)) ]
    return outliers

outliers = np.array(outlier_via_ID_matrix(x_train, y_train))
# outliers = np.array(outlier_via_svm(x_train, y_train)) # --> Performs worse than the other one!

indexes = [i for i in range(len(outliers)) if outliers[i] == 1]
# print(indexes)
# Remove the outliers from the data
x_train_filtered = x_train[x_train.index.isin(indexes)]
y_train_filtered = y_train[y_train.index.isin(indexes)]

# Number of selected samples
print("Total number of samples: ", len(x_train))
print("Number of 'outliers' removed: ", len(x_train)-len(indexes))



# Testing with simple SVR regression  --> Surprisingly a very good result using this!!! No idea why!!
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import time

use_filtered_data =True

if use_filtered_data:
    X = x_train_filtered
    y = y_train_filtered
else:
    X = pd.read_csv(ROOT_PATH + 'X_train.csv').drop("id", axis=1)
    y = pd.read_csv(ROOT_PATH + 'y_train.csv').drop("id", axis=1)["y"]

# Remove zero variance features
temp = x_train.nunique(axis=0)
temp = temp[temp == 1]  # constant features
const_features = temp.index.values.tolist()
x_train = x_train.drop(const_features, axis=1)

pipe_pre = Pipeline([
    ('s1', SimpleImputer(strategy='median')),
    ('s2', MinMaxScaler()),
    ('s4', SelectKBest(score_func=f_regression)),
    ('s5', SVR(kernel="rbf"))
])
grid = {  # Took 12 min to do the optimization (4000 options)
    's4__k': [75], # np.linspace(60, 110, 20, dtype=int),
    's5__C': [100], # np.linspace(1,200, 50),
    's5__gamma': ['scale'],
    's5__epsilon': [2.222],
}

tic = time.time()
estimator = GridSearchCV(pipe_pre, grid, cv=5, n_jobs=16, scoring="r2").fit(X, y)
model = estimator.best_estimator_
params = estimator.best_params_
scores = cross_val_score(model, X, y, cv=5, n_jobs=16, scoring="r2")
scores = scores.flatten()
print("Best parameters: ", params)
print("Mean: ", scores.mean())
print("Std: ", scores.std())
toc = time.time()
print(toc-tic, " seconds")


# Output the predictions for the test data
X_test = pd.read_csv(ROOT_PATH + "X_test.csv")
print(const_features)
X_test = X_test.drop(const_features, axis=1)
final = model.fit(X, y)
pd.DataFrame(final.predict(X_test)).to_csv(ROOT_PATH+"AML_task1_v2.csv", index_label='id', header=['y'])
