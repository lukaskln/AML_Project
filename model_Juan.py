# Outlier removal in our pipeline!!
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

ROOT_PATH = "/home/jupyter/Task 1/"


def outlier_rejection(X, y, contamination=0.05, nu=0.05, max_features=1.0):
    """This will be our function used to resample our dataset."""
    model = OneClassSVM(nu=nu)
    y_pred = model.fit_predict(X)
    print(len(X[y_pred == 1]), len(X))
    return X[y_pred == 1], y[y_pred == 1]


tic = time.time()
X = pd.read_csv(ROOT_PATH + 'X_train.csv').drop("id", axis=1)
y = pd.read_csv(ROOT_PATH + 'y_train.csv').drop("id", axis=1)["y"]

# Remove zero variance features
temp = X.nunique(axis=0)
temp = temp[temp == 1]  # constant features
const_features = temp.index.values.tolist()
X = X.drop(const_features, axis=1)

# Impute Nan values (needed before we can run the OneClassSVM to remove outliers)
imp = SimpleImputer(strategy="median")
X = imp.fit_transform(X)

# Nu == 0.05, was picked based on CV scores on the train data (for each nu value I did param tuning of the SVR model)
X, y = outlier_rejection(X, y, nu=0.05)


pipe_pre = Pipeline([
    ('s1', SimpleImputer(strategy='median')),
    ('s2', QuantileTransformer(output_distribution="normal", random_state=42)),
    ('s4', SelectKBest(score_func=f_regression)),
    ('s5', SVR(kernel="rbf"))
])
grid = {  # Took 12 min to do the optimization (4000 options)
    's4__k': [201], # np.linspace(201, 201, 1, dtype=int),  # np.linspace(60, 300, 100, dtype=int), 
    's5__C': [61.1], # np.linspace(60,70, 10), 
    's5__gamma': ['auto'], # ['scale', 'auto'],
    's5__epsilon': [0],  # np.linspace(0,1, 5)   # Any CV I was doing 0, was the value chosen, so I jsut fixed it now to tune the others better. 
}

estimator1 = GridSearchCV(pipe_pre, grid, cv=5, n_jobs=16, scoring="r2", verbose=2).fit(X, y)
# %store estimator1

model = estimator1.best_estimator_
params = estimator1.best_params_
scores = cross_val_score(model, X, y, cv=5, n_jobs=16, scoring="r2")
scores = scores.flatten()
print(0.05, scores.mean(), params)


# Save the GridSearch outputs
# logs = estimator1.cv_results_
# df = pd.DataFrame.from_dict(logs)
# df.to_csv(ROOT_PATH + "Logs_Juan_model_tuning.csv", index=False)

# Time taken to train
toc = time.time()
print(toc-tic, " seconds")

# Run the precition on the test data
X_test = pd.read_csv(ROOT_PATH + "X_test.csv").drop("id", axis=1)
X_test= X_test.drop(const_features, axis=1)
pd.DataFrame(estimator1.best_estimator_.predict(X_test)).to_csv("AML_task1_optimized_Juan_sol.csv", index_label='id', header=['y'])

# Performance:
# Train: 0.668
# Test: 0.738



