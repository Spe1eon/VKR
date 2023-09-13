import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler

dailyData = pd.read_csv("Adidas Sales/sales_simplified.csv")
dailyData.drop("Unnamed: 0", axis=1, inplace=True)
dailyData.sort_values(by="index", inplace=True)
dailyData["index"] = pd.to_datetime(dailyData["index"])
weeklyData = pd.DataFrame()
for product in dailyData["Product"].unique():
  buffer = dailyData[dailyData["Product"] == product].copy()
  buffer.sort_values(by="index", inplace=True)
  buffer = buffer.resample("W", kind="timestamp", on="index").mean()
  buffer["Product"] = product
  buffer.reset_index(level="index", inplace=True)
  if len(weeklyData) == 0:
    weeklyData = buffer
  else:
    weeklyData = pd.concat([weeklyData, buffer])

dailyData["dayofweek"] = dailyData["index"].dt.dayofweek
dailyData["quarter"] = dailyData["index"].dt.quarter
dailyData["month"] = dailyData["index"].dt.month
dailyData["year"] = dailyData["index"].dt.year
dailyData["dayofyear"] = dailyData["index"].dt.dayofyear
dailyData["dayofmonth"] = dailyData["index"].dt.day
dailyData["weekofyear"] = dailyData["index"].dt.weekofyear

weeklyData["quarter"] = weeklyData["index"].dt.quarter
weeklyData["month"] = weeklyData["index"].dt.month
weeklyData["year"] = weeklyData["index"].dt.year
weeklyData["weekofyear"] = weeklyData["index"].dt.weekofyear

dailyTreshold = "2021-11-01"
weeklyTreshold = "2021-10-24"

dailyData_train, dailyData_test = dailyData.loc[dailyData["index"] <= pd.Timestamp(dailyTreshold)].copy(), dailyData.loc[dailyData["index"] > pd.Timestamp(dailyTreshold)].copy()

weeklyData_train, weeklyData_test = weeklyData.loc[weeklyData["index"] <= pd.Timestamp(weeklyTreshold)].copy(), weeklyData.loc[weeklyData["index"] > pd.Timestamp(weeklyTreshold)].copy()

def getValues(df, label):
  buffer = df[df["Product"] == label].drop(["Product","index"], axis=1)
  X, Y = buffer.drop("Units Sold", axis=1), buffer["Units Sold"]
  return X, Y

def objectiveDaily(trial):
  Xtrain, Ytrain = getValues(dailyData_train, product)
  Xtest, Ytest = getValues(dailyData_test, product)

  param_grid = {
      'max_depth': trial.suggest_int('max_depth', 6, 10),
      'n_estimators': trial.suggest_int('n_estimators', 400, 4000, 400),
      'eta': trial.suggest_float('eta', 0.007, 0.013),
      'tree_method': trial.suggest_categorical('tree_method', ['exact', 'approx', 'hist']),
      'subsample': trial.suggest_discrete_uniform('subsample', 0.2, 0.9, 0.1),
      'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.2, 0.9, 0.1),
      'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.2, 0.9, 0.1),
      'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 1e4),
      'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e4),
      'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e4),
      'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e4),
  } 
  
  reg = xgb.XGBModel(
      objective='reg:squarederror',
      **param_grid
  )
  
  reg.fit(Xtrain, Ytrain,
          eval_set=[(Xtrain, Ytrain), (Xtest, Ytest)], eval_metric='rmse',
          verbose=False)

  return mean_squared_error(Ytest, reg.predict(Xtest), squared=False)

dailyRegressorsParams = {}
# train_time = 3600
train_time = 60 * 10
for product in dailyData["Product"].unique():
  study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='XGBRegressor')
  study.optimize(objectiveDaily, timeout=train_time, show_progress_bar=True)
  trial = study.best_trial
  dailyRegressorsParams[product] = trial.params

pd.DataFrame(dailyRegressorsParams, index=dailyRegressorsParams[product].keys())

dailyRegressors = {}
for product in dailyRegressorsParams.keys():
  Xtrain, Ytrain = getValues(dailyData_train, product)
  Xtest, Ytest = getValues(dailyData_test, product)
  dailyRegressor = xgb.XGBRegressor(early_stopping_rounds=50, eval_metric="rmse", **dailyRegressorsParams[product])
  dailyRegressor.fit(Xtrain, Ytrain,
                    eval_set=[(Xtrain, Ytrain), (Xtest, Ytest)],
                    verbose=False)
  dailyRegressors[product] = dailyRegressor

# Важность параметров
fig, axs = plt.subplots(6, 1, figsize=(20, 35))
for i, product in enumerate(dailyRegressors.keys()):
  _ = plot_importance(dailyRegressors[product], ax=axs[i], height=0.9)
  axs[i].set_title(product)

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(dailyData_train["Product"].unique()):
  test_buffer = dailyData_test[dailyData_test["Product"] == product].copy()
  train_buffer = dailyData_train[dailyData_train["Product"] == product].copy()

  test_buffer["Predictions"] = dailyRegressors[product].predict(test_buffer.drop(["Product", "index", "Units Sold"], axis=1))

  bufferAll = pd.concat([train_buffer, test_buffer])

  axs[i].plot(bufferAll["index"], bufferAll["Units Sold"], label="Raw")
  axs[i].plot(bufferAll["index"], bufferAll["Predictions"], label="Predicted")

  axs[i].set_title(product)
  axs[i].grid()

for i, product in enumerate(dailyData_train["Product"].unique()):
  test_buffer = dailyData_test[dailyData_test["Product"] == product].copy()
  error = mean_squared_error(y_true=test_buffer['Units Sold'], y_pred=dailyRegressors[product].predict(test_buffer.drop(["Product", "index", "Units Sold"], axis=1)), squared=False)
  print("Product: {} Error: {}".format(product, error))

def objectiveWeekly(trial):
  Xtrain, Ytrain = getValues(weeklyData_train, product)
  Xtest, Ytest = getValues(weeklyData_test, product)

  param_grid = {
      'max_depth': trial.suggest_int('max_depth', 6, 10),
      'n_estimators': trial.suggest_int('n_estimators', 400, 4000, 400),
      'eta': trial.suggest_float('eta', 0.007, 0.013),
      'tree_method': trial.suggest_categorical('tree_method', ['exact', 'approx', 'hist']),
      'subsample': trial.suggest_discrete_uniform('subsample', 0.2, 0.9, 0.1),
      'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.2, 0.9, 0.1),
      'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.2, 0.9, 0.1),
      'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 1e4),
      'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e4),
      'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e4),
      'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e4),
  } 
  
  reg = xgb.XGBModel(
      objective='reg:squarederror',
      **param_grid
  )
  
  reg.fit(Xtrain, Ytrain,
          eval_set=[(Xtrain, Ytrain), (Xtest, Ytest)], eval_metric='rmse',
          verbose=False)

  return mean_squared_error(Ytest, reg.predict(Xtest), squared=False)

weeklyRegressorsParams = {}
# train_time = 3600
train_time = 60 * 10
for product in dailyData["Product"].unique():
  study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='XGBRegressor')
  study.optimize(objectiveWeekly, timeout=train_time, show_progress_bar=True)
  trial = study.best_trial
  weeklyRegressorsParams[product] = trial.params

pd.DataFrame(weeklyRegressorsParams, index=weeklyRegressorsParams[product].keys())

weeklyRegressors = {}
for product in weeklyRegressorsParams.keys():
  Xtrain, Ytrain = getValues(weeklyData_train, product)
  Xtest, Ytest = getValues(weeklyData_test, product)
  weeklyRegressor = xgb.XGBRegressor(early_stopping_rounds=50, eval_metric="rmse", **weeklyRegressorsParams[product])
  weeklyRegressor.fit(Xtrain, Ytrain,
                    eval_set=[(Xtrain, Ytrain), (Xtest, Ytest)],
                    verbose=False)
  weeklyRegressors[product] = weeklyRegressor

# Важность параметров
fig, axs = plt.subplots(6, 1, figsize=(20, 35))
for i, product in enumerate(weeklyRegressors.keys()):
  _ = plot_importance(weeklyRegressors[product], ax=axs[i], height=0.9)
  axs[i].set_title(product)

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(weeklyData_train["Product"].unique()):
  test_buffer = weeklyData_test[weeklyData_test["Product"] == product].copy()
  train_buffer = weeklyData_train[weeklyData_train["Product"] == product].copy()

  test_buffer["Predictions"] = weeklyRegressors[product].predict(test_buffer.drop(["Product", "index", "Units Sold"], axis=1))

  bufferAll = pd.concat([train_buffer, test_buffer])

  axs[i].plot(bufferAll["index"], bufferAll["Units Sold"], label="Raw")
  axs[i].plot(bufferAll["index"], bufferAll["Predictions"], label="Predicted")

  axs[i].set_title(product)
  axs[i].grid()

for i, product in enumerate(weeklyData_train["Product"].unique()):
  test_buffer = weeklyData_test[weeklyData_test["Product"] == product].copy()
  error = mean_squared_error(y_true=test_buffer['Units Sold'], y_pred=weeklyRegressors[product].predict(test_buffer.drop(["Product", "index", "Units Sold"], axis=1)))
  print("Product: {} Error: {}".format(product, error))





