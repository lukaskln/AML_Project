#### SetUp

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.svm import SVR, OneClassSVM
from sklearn.decomposition import PCA
import xgboost as xgb

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

from collections import Counter
from scipy import stats
from itertools import repeat


#### Preprocessing

# Train Data

X_train = pd.read_csv("X_train.csv")

y_train = pd.read_csv("y_train.csv")

X_train.fillna(X_train.median(), inplace=True)

X_train = X_train.iloc[:, 1:833]

scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(
        X_train),
    columns=X_train.columns)

# Test Data

X_test = pd.read_csv("X_test.csv")

X_test.fillna(X_test.median(), inplace=True)

X_test = X_test.iloc[:, 1:833]

X_test = pd.DataFrame(
    scaler.fit_transform(
        X_test),
    columns=X_test.columns)

#### Feature Selection

# Lasso Based

lsvc = Lasso(alpha=0.3).fit(X_train, y_train["y"])
model = SelectFromModel(lsvc, prefit=True)

X_train = pd.DataFrame(model.transform(X_train))
X_test = pd.DataFrame(model.transform(X_test))

# Correlation Based

Corr = X_train.corrwith(y_train["y"], method="pearson")

X_train = X_train.loc[:, (Corr > 0.1) | (Corr < -0.1)]
X_test = X_test.loc[:, (Corr > 0.1) | (Corr < -0.1)]

#### Outlier correction

pca = PCA(n_components=20)
pca.fit(X_train)
dfComp = pd.DataFrame(pca.transform(X_train))

outliers_fraction = 0.001
nu_estimate = 0.95 * outliers_fraction + 0.04
auto_detection = OneClassSVM(kernel="rbf", gamma=0.00001, nu=nu_estimate)
auto_detection.fit(dfComp.iloc[:, [0, 1]])
evaluation = auto_detection.predict(dfComp.iloc[:, [0, 1]])

X_train = X_train[evaluation != -1]
y_train = y_train[evaluation != -1]

#### Model

# Grid Search

param_test = {'max_depth': [20, 19, 18],
              'min_child_samples': [5, 6, 7],
              'min_child_weight': [0],
              'subsample': [0.8, 0.85],
              'colsample_bytree': [0.75, 0.7]}

gsearch5 = GridSearchCV(estimator=lgb.LGBMRegressor(boosting_type='gbdt',
                                                    objective='rmse',
                                                    num_iterations=800,
                                                    learning_rate=0.07,
                                                    metric='l1',
                                                    num_threads=2,
                                                    random_state=42),
                        param_grid=param_test,
                        scoring="r2",
                        n_jobs=4,
                        cv=5)

gsearch5.fit(X_train, y_train["y"])

pd.DataFrame(gsearch5.cv_results_).sort_values("rank_test_score")
print(gsearch5.best_params_)
print(gsearch5.best_score_)

# Finetuning

R2 = []
cv = KFold(n_splits=5, random_state=42)

model = lgb.LGBMRegressor(boosting_type='gbdt',
                          objective='rmse',
                          num_iterations=800,
                          learning_rate=0.07,
                          metric='l1',
                          random_state=42,
                          max_depth=19,
                          min_child_samples=6,
                          min_child_weight=0,
                          subsample=0.8,
                          colsample_bytree=0.7,
                          reg_alpha=0,
                          importance_type="split"
                          )

for train_ix, test_ix in cv.split(X_train):

    X_cvtrain, X_cvtest = X_train.iloc[train_ix, :], X_train.iloc[test_ix, :]
    y_cvtrain, y_cvtest = y_train["y"].iloc[train_ix], y_train["y"].iloc[test_ix]

    model.fit(X_cvtrain, y_cvtrain)

    predtrain = model.predict(X_cvtrain)
    pred = model.predict(X_cvtest)

    print("\nTrain R2:")
    print(np.round(r2_score(y_cvtrain, predtrain), 2))
    print("\nTest R2:")
    print(np.round(r2_score(y_cvtest, pred), 2))
    print("\n________________________")

    R2.append(np.round(r2_score(y_cvtest, pred), 4))

print("\nAverage R2:", round(np.sum(R2)/5, 2))
print("Std:", round(np.std(R2), 4))


# Predict Test Data

model.fit(X_train, y_train["y"])

preds = model.predict(X_test)

dfResults = pd.DataFrame({"id": list(range(0, 776, 1)), "y": preds})

dfResults.to_csv("Results.csv", sep=',', index=False)
