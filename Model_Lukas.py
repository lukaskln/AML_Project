# SetUp

from matplotlib.pylab import rcParams
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
from autoreject import set_matplotlib_defaults  # noqa
set_matplotlib_defaults(plt)

rcParams['figure.figsize'] = 15, 4



def modelfit(alg, X , y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)

    alg.fit(X, y, eval_metric='rmse')

    dtrain_predictions = alg.predict(X)

    print("\nModel Report:")
    print("\nTrain R^2 : %.4g" % r2_score(y, dtrain_predictions))
    print("\nLast Iteration:")
    print(cvresult.iloc[-1,:])
    print("\nBest Iteration:")
    print(cvresult[cvresult["test-rmse-mean"] ==
                   np.min(cvresult["test-rmse-mean"])])

    param_range = np.linspace(1, [x for x in cvresult["test-rmse-mean"].shape][0], [x for x in cvresult["test-rmse-mean"].shape][0])

    plt.figure(figsize=(8, 5))
    plt.tick_params(axis='x', which='both', bottom='off', top='off')
    plt.tick_params(axis='y', which='both', left='off', right='off')

    colors = ["#049DD9", "#03A64A", "#F2AC29", "#F2CA80", "#F22929"]

    plt.plot(param_range, cvresult["test-rmse-mean"],
            'o-', markerfacecolor='w',
            color=colors[0], markeredgewidth=2, linewidth=2,
            markeredgecolor=colors[0], markersize=8, label='CV score')
    plt.axvline(cvresult[cvresult["test-rmse-mean"] == np.min(cvresult["test-rmse-mean"])].index[0], label='Minimum', color=colors[2],
                linewidth=2, linestyle='--')
    plt.ylabel('RMSE')
    plt.xlabel('Interation')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    


# Choose all predictors except target & IDcols
# Fix learning rate and number of estimators for tuning tree-based parameters


xgb1 = XGBRegressor(
    objective= "reg:squarederror",
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    min_child_weight=5,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=4,
    scale_pos_weight=1,
    seed=27)
    
modelfit(xgb1, X = X_train , y = y_train["y"])

# 1. Tune max_depth and min_child_weight

param_test1 = {
    'max_depth': range(3, 8, 1),
    'min_child_weight': range(3, 8, 1)
}

gsearch1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=250, max_depth=6,
                                                min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective="reg:squarederror", nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test1, scoring="r2", n_jobs=4, iid=False, cv=5)
gsearch1.fit(X_train, y_train["y"])
pd.DataFrame(gsearch1.cv_results_).sort_values("rank_test_score").iloc[:,[4,5,12,13,14]] # 5,6 best

# 2. Tune Gamma

param_test2 = {
    'gamma': [i/10.0 for i in range(0, 5)]
}
gsearch2 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=250, max_depth=6,
                                                min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective="reg:squarederror", nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test2, scoring="r2", n_jobs=4, iid=False, cv=5)
gsearch2.fit(X_train, y_train["y"])
pd.DataFrame(gsearch2.cv_results_).sort_values("rank_test_score") # 0.1 best

# Recalibrate Boosting rounds

xgb2 = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.07,
    n_estimators=200,
    max_depth=6,
    min_child_weight=5,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=4,
    scale_pos_weight=1,
    seed=27)

modelfit(xgb2, X=X_train, y=y_train["y"]) #500

# 3. Tune subsample and colsample_bytree

param_test3 = {
    'subsample': [i/10.0 for i in range(6, 10)],
    'colsample_bytree': [i/10.0 for i in range(6, 10)]
}
gsearch3 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.07, n_estimators=200, max_depth=6, gamma=0.1,
                                                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                                objective="reg:squarederror", nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test3, scoring="r2", n_jobs=4, iid=False, cv=5)
gsearch3.fit(X_train, y_train["y"])
pd.DataFrame(gsearch3.cv_results_).sort_values("rank_test_score") # 0.7, 0.7

# 4. Tuning Regularization Parameters

param_test4 = {
    'reg_alpha': [0.07, 0.08, 0.1, 0.12, 0.15]
}
gsearch4 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.07, n_estimators=200, max_depth=6,
                                                min_child_weight=5, gamma=0.1, subsample=0.7, colsample_bytree=0.7,
                                                objective="reg:squarederror", nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test4, scoring="r2", n_jobs=4, iid=False, cv=5)
gsearch4.fit(X_train, y_train["y"])
pd.DataFrame(gsearch4.cv_results_).sort_values("rank_test_score") # 0.1

# Tune learning rate / n_estimators
# Selfcoded CV

# 5 Fold CV


R2 = []
cv = KFold(n_splits=5, random_state=42, shuffle=True)

xgb4 = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.07,
    n_estimators=500,
    max_depth=6,
    min_child_weight=5,
    gamma=0.2,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha = 0.08,
    nthread=4,
    seed=42)

for train_ix, test_ix in cv.split(X_train):

    X_cvtrain, X_cvtest = X_train.iloc[train_ix, :], X_train.iloc[test_ix, :]
    y_cvtrain, y_cvtest = y_train["y"].iloc[train_ix], y_train["y"].iloc[test_ix]

    xgb4.fit(X_cvtrain, y_cvtrain)

    predtrain = xgb4.predict(X_cvtrain)
    pred = xgb4.predict(X_cvtest)

    print("\nTrain R2:")
    print(np.round(r2_score(y_cvtrain, predtrain), 2))
    print("\nTest R2:")
    print(np.round(r2_score(y_cvtest, pred),2))
    print("\n________________________")

    R2.append(np.round(r2_score(y_cvtest, pred), 4))

print("\nAverage R2:",round(np.sum(R2)/5,2))
print("Std:", round(np.std(R2),4))

# Predict Test Data

xgb4.fit(X_train, y_train["y"])


X_test.fillna(X_train.median(), inplace=True)

X_test = X_test.iloc[:, 1:833]

# Scaling / Normalization
scaler = StandardScaler()  # Play around

X_test = pd.DataFrame(
    scaler.fit_transform(
        X_test),
    columns=X_test.columns)

X_test = X_test.loc[:, (Corr > 0.1) | (Corr < -0.1)] # Corr from EDA

preds = xgb4.predict(X_test)


dfResults = pd.DataFrame({"id": list(range(0,776,1)), "y": preds})

dfResults.to_csv("Results.csv", sep=',',index = False)

