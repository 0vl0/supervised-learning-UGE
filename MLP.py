"""
Multilayer Perceptron (MLP) script for Gustave Eiffel University.
See MLP.ipynb for comments.
"""

from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description="MLP model to predict mechanical parameters")

parser.add_argument("--y","-y", type=str, default="Rc",
  help="y: mechanical parameter to predict")

parser.add_argument("--excel_path","-lp", type=str, default='data/data_v2.xlsx',
  help="Where to load excel file")

parser.add_argument("--excel_sheet","-sh", type=str, default='7-90j',
  help="Excel sheet to load")

parser.add_argument("--save_path","-sp", type=str, default='results/MLP',
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
    f.write("MLP model\n\nFeatures: {}\nTarget: {}\n".format(column_X, column_Y))

train, test = train_test_split(df, test_size=0.5)
X_train, X_val = train[column_X], test[column_X]
y_train, y_val = train[column_Y], test[column_Y]
std_scaler = StandardScaler()
y_train_scaled = std_scaler.fit_transform(y_train)
y_test_scaled = std_scaler.transform(y_val)

mlp_regressor = MLPRegressor(activation="logistic", solver="lbfgs", max_iter=500)
mlp_pipe = make_pipeline(StandardScaler(), mlp_regressor)
H_grid = np.arange(8, 400, step=1)
param_grid = {"mlpregressor__hidden_layer_sizes":H_grid}
grid_search = GridSearchCV(mlp_pipe, param_grid=param_grid)
grid_search.fit(X_train, y_train_scaled.ravel())
hls = grid_search.best_params_["mlpregressor__hidden_layer_sizes"]

with open("{}/{}/results.txt".format(SAVE_PATH, TARGET), 'a') as f:
    f.write("Number of hidden layers: = {}\n\n".format(hls))
    f.write("Training score: {}\n".format(grid_search.best_score_))

MLP = make_pipeline(StandardScaler(), MLPRegressor(solver="lbfgs", hidden_layer_sizes=hls, max_iter=1000))
MLP.fit(X_train, y_train_scaled.ravel())

y_pred = MLP.predict(X_val)
score_val = MLP.score(X_val, std_scaler.transform(y_val))
with open("{}/{}/results.txt".format(SAVE_PATH, TARGET), 'a') as f:
    f.write("Validation score: {}\n".format(score_val))

pickle.dump(MLP, open("{}/{}/MLP_{}_model".format(SAVE_PATH,TARGET,TARGET), "wb"))

r_train = permutation_importance(MLP, X_train, y_train_scaled, n_repeats=100)
features = np.array(column_X)
sorted_idx_train = r_train.importances_mean.argsort()
plt.figure(figsize=(11,5))
plt.barh(features[sorted_idx_train], r_train.importances_mean[sorted_idx_train])
plt.xlabel("Permutation Importance", fontsize=14)
plt.title("Importance of features of the training set", fontsize=18)
plt.savefig("{}/{}/permutation_importance_train.png".format(SAVE_PATH, TARGET))

r_val = permutation_importance(MLP, X_val, std_scaler.transform(y_val), n_repeats=30)
sorted_idx = r_val.importances_mean.argsort()
plt.figure(figsize=(11,5))
plt.barh(features[sorted_idx], r_val.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance", fontsize=14)
plt.title("Importance of features of the validation set", fontsize=18)
plt.savefig("{}/{}/permutation_importance_val.png".format(SAVE_PATH, TARGET))

def f(x):
    return x 
y_pred_scaled = std_scaler.inverse_transform(y_pred.reshape(-1,1))
bound_sup = max(y_val.max()[TARGET], y_pred_scaled.max())
bound_inf = min(y_val.min()[TARGET], y_pred_scaled.min())
X = np.linspace(bound_inf-.2,bound_sup+.2,100)
plt.figure(figsize=(7,7))
plt.scatter(y_val, y_pred_scaled, c="blue")
plt.plot(X,f(X), c="gray")
plt.title("MLP model - {} predicted against {} validation".format(TARGET, TARGET), fontsize=18, fontweight="bold")
plt.xlabel("{}_val ({})\n".format(TARGET, unit[TARGET]), fontsize=14)
plt.ylabel("{}_pred ({})".format(TARGET, unit[TARGET]), fontsize=14)
plt.xlim(left=bound_inf-.2, right=bound_sup+.2)
plt.ylim(bottom=bound_inf-.2, top=bound_sup+.2)
plt.grid()
plt.savefig("{}/{}/results.png".format(SAVE_PATH, TARGET))