"""
SVR script for Gustave Eiffel University.
See svm.ipynb for comments.
"""

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import neighbors
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
import numpy as np
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description="SVM model to predict mechanical parameters")

parser.add_argument("--y","-y", type=str, default="Rc",
  help="y: mechanical parameter to predict")

parser.add_argument("--excel_path","-lp", type=str, default='data/data_v2.xlsx',
  help="Where to load excel file")

parser.add_argument("--excel_sheet","-sh", type=str, default='7-90j',
  help="Excel sheet to load")

parser.add_argument("--save_path","-sp", type=str, default='results/SVR',
  help="Where to save results")

args = parser.parse_args()
TARGET = args.y
EXCEL_PATH = args.excel_path
SAVE_PATH = args.save_path
EXCEL_SHEET= args.excel_sheet

unit = {"Vp": "m/s", "Rc": "MPa", "e50_local":"MPa"}

if not os.path.exists(SAVE_PATH+"/"+TARGET): 
    print("creating save directory !")
    os.makedirs(SAVE_PATH+"/"+TARGET)

"""
Load data from excel
"""
df = pd.read_excel(EXCEL_PATH, EXCEL_SHEET)
column_X = ["t_cure", "teneur_ciment", "w_t", "silt", "kaol", "s", "illite", "W/C"]
column_Y = [TARGET]
X = df[column_X]
y = df[column_Y]
with open("{}/{}/results.txt".format(SAVE_PATH, TARGET), 'w') as f:
    f.write("SVR model\n\nFeatures: {}\nTarget: {}\n".format(column_X, column_Y))

std_scaler = StandardScaler()
y_scaled = std_scaler.fit_transform(y)
n_neighbors = 3
knn = neighbors.KNeighborsRegressor(n_neighbors, weights="uniform")
knn_pipe = make_pipeline(StandardScaler(), knn)
y_ = knn_pipe.fit(X,y_scaled).predict(X)
n = len(y)
sigma = sqrt(3*n**(1/5)/(3*n-1))*norm(y_-y_scaled)
epsilon =  sigma/sqrt(n)

train, test = train_test_split(df, test_size=0.5)
X_train, X_val = train[column_X], test[column_X]
y_train, y_val = train[column_Y], test[column_Y]
std_scaler2 = StandardScaler()
y_train_scaled = std_scaler2.fit_transform(y_train)
mean_y = y_train.mean()
sigma_y = y_train.std()
C = max(abs(mean_y+3*sigma_y)[0],abs(mean_y-3*sigma_y)[0])

knn = neighbors.KNeighborsRegressor(n_neighbors, weights="uniform")
SVR_pipe = make_pipeline(StandardScaler(), svm.SVR(kernel="rbf", C=C, epsilon=epsilon))

gamma_grid = np.arange(1e-4, 1e2, step=0.1)
param_grid = {"svr__gamma":gamma_grid}
grid_search = GridSearchCV(SVR_pipe, param_grid=param_grid)
grid_search.fit(X_train, y_train_scaled.ravel())

with open("{}/{}/results.txt".format(SAVE_PATH, TARGET), 'a') as f:
    f.write("\nParameters chosen\nC = {}\nEpsilon = {}\nGamma = {}\n\n".format(C, epsilon, grid_search.best_params_["svr__gamma"]))
    f.write("Training score: {}\n".format(grid_search.best_score_))

gamma = grid_search.best_params_["svr__gamma"]
SVR = make_pipeline(StandardScaler(), svm.SVR(kernel="rbf", gamma=gamma, epsilon=epsilon, C=C))
SVR.fit(X_train, y_train_scaled.ravel())

y_pred = SVR.predict(X_val)
score_val = SVR.score(X_val, std_scaler2.transform(y_val))
with open("{}/{}/results.txt".format(SAVE_PATH, TARGET), 'a') as f:
    f.write("Validation score: {}\n".format(score_val))

pickle.dump(SVR, open("{}/{}/SVR_{}_model".format(SAVE_PATH,TARGET,TARGET), "wb"))

r_train = permutation_importance(SVR, X_train, y_train_scaled, n_repeats=30)
features = np.array(column_X)
sorted_idx_train = r_train.importances_mean.argsort()
plt.figure(figsize=(11,5))
plt.barh(features[sorted_idx_train], r_train.importances_mean[sorted_idx_train])
plt.xlabel("Permutation Importance", fontsize=14)
plt.title("Importance of features of the training set", fontsize=18)
plt.savefig("{}/{}/permutation_importance_train.png".format(SAVE_PATH, TARGET))

r_val = permutation_importance(SVR, X_val, std_scaler2.transform(y_val), n_repeats=30)
sorted_idx = r_val.importances_mean.argsort()
plt.figure(figsize=(11,5))
plt.barh(features[sorted_idx], r_val.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance", fontsize=14)
plt.title("Importance of features of the validation set", fontsize=18)
plt.savefig("{}/{}/permutation_importance_val.png".format(SAVE_PATH, TARGET))

def f(x):
    return x 
y_pred_scaled = std_scaler2.inverse_transform(y_pred.reshape(-1,1))
bound_sup = max(y_val.max()[TARGET], y_pred_scaled.max())
bound_inf = min(y_val.min()[TARGET], y_pred_scaled.min())
X = np.linspace(bound_inf-.2,bound_sup+.2,100)
plt.figure(figsize=(7,7))
plt.scatter(y_val, y_pred_scaled, c="blue")
plt.plot(X,f(X), c="gray")
plt.title("SVR model - {} predicted against {} validation".format(TARGET, TARGET), fontsize=18, fontweight="bold")
plt.xlabel("{}_val ({})\n".format(TARGET, unit[TARGET]), fontsize=14)
plt.ylabel("{}_pred ({})".format(TARGET, unit[TARGET]), fontsize=14)
plt.xlim(left=bound_inf-.2, right=bound_sup+.2)
plt.ylim(bottom=bound_inf-.2, top=bound_sup+.2)
plt.grid()
plt.savefig("{}/{}/results.png".format(SAVE_PATH, TARGET))