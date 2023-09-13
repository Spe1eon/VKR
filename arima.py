import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import optuna
import math
import pickle as pkl
import warnings
''' При обучении очень часто возникает следующее предупреждение "/usr/local/lib/python3.8/dist-packages/statsmodels/tsa/base/tsa_model.py:132: FutureWarning: The 'freq' argument in Timestamp is deprecated 
and will be removed in a future version. date_key = Timestamp(key, freq=base_index.freq)" Блокируем ее!'''
warnings.filterwarnings("ignore", category=FutureWarning)

rawData = pd.read_csv("Adidas Sales/sales_simplified.csv")
rawData.drop("Unnamed: 0", axis=1, inplace=True)
rawData.sort_values(by="index", inplace=True)

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(rawData["Product"].unique()):
  Y = rawData[rawData["Product"] == product]["Units Sold"]
  sm.graphics.tsa.plot_acf(Y, lags=200, title=product, ax=axs[i])
  axs[i].grid()

ACFResult = {
    "Men's Street Footwear": 5, 
    "Women's Athletic Footwear": 3, 
    "Women's Street Footwear": 6, 
    "Women's Apparel": 1, 
    "Men's Athletic Footwear": 3, 
    "Men's Apparel": 1
}

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(rawData["Product"].unique()):
  Y = rawData[rawData["Product"] == product]["Units Sold"]
  sm.graphics.tsa.plot_pacf(Y, lags=150, title=product, ax=axs[i])
  axs[i].grid()

PACFResult = {
    "Men's Street Footwear": 7, 
    "Women's Athletic Footwear": 6, 
    "Women's Street Footwear": 6, 
    "Women's Apparel": 13, 
    "Men's Athletic Footwear": 9, 
    "Men's Apparel": 10
}

# Класс-обертка
class ARIMAEstimator:
  def __init__(self, train, test, p_max, d_max, q_max):
    self.train = train
    self.test = test
    self.p_max = p_max if p_max > 0 else 1
    self.d_max = d_max if d_max > 0 else 1
    self.q_max = q_max if q_max > 0 else 1
  
  def __call__(self, trial):
    p_value = trial.suggest_int("p_value", 1, self.p_max)
    d_value = trial.suggest_int("d_value", 0, self.d_max)
    q_value = trial.suggest_int("q_value", 1, self.q_max)
    model = ARIMA(self.train, order=(p_value, d_value, q_value))
    result = model.fit()
    pred = result.get_prediction(self.test.index.min(), self.test.index.max()).predicted_mean
    return mean_squared_error(self.test, pred, squared=False)

# Функции, чтобы минмизировать повторение кода
def trainTestSplit(data, product, treshold, freq="D"):
  buffer = data[data["Product"] == product].copy().drop("Product", axis=1)
  buffer.index = pd.DatetimeIndex(buffer["index"].values, freq="D")
  if freq != "D":
    buffer = buffer.resample(freq, kind="timestamp").mean()  
  buffer = buffer["Units Sold"]
  train, test = buffer[: treshold], buffer[treshold :]
  return train, test

def performEstiation(train, test, p_max, d_max, q_max):
  estimator = ARIMAEstimator(train, test, p_max, d_max, q_max)
  study = optuna.create_study(direction="minimize")
  study.optimize(estimator, n_trials=20)
  p_value, d_value, q_value = study.best_params["p_value"], study.best_params["d_value"], study.best_params["q_value"]
  rmse = study.best_value
  return {"RMSE":rmse, "(p,d,q)":(p_value, 0, q_value)}

bestModels = {}
treshold = "2021-11-01"
for product in rawData["Product"].unique():
  if len(bestModels) > 0:
    with open("drive/MyDrive/Adidas Sales/ARIMA_best_daily_usual.pkl", "wb") as file:
      pkl.dump(bestModels, file)
  train, test = trainTestSplit(rawData, product, treshold)
  bestParams = performEstiation(train, test, ACFResult[product], 0, PACFResult[product])
  if (product not in bestModels) or (bestParams["RMSE"] < bestModels[product]["RMSE"]):
    print("Product: {} (p,d,q): {} rmse: {}".format(product, bestParams["(p,d,q)"], bestParams["RMSE"]))
    bestModels[product] = bestParams

for product in bestModels:
  print("Product: {} (p,d,q): {} rmse: {}".format(product, bestModels[product]["(p,d,q)"], bestModels[product]["RMSE"]))

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(rawData["Product"].unique()):
  buffer = rawData[rawData["Product"] == product].copy()
  buffer["index"] =  pd.to_datetime(buffer["index"])
  buffer = buffer.sort_values(by="index").set_index("index").drop("Product", axis=1)
  buffer = buffer.resample("W", kind="timestamp").mean().diff().dropna()
  sm.graphics.tsa.plot_acf(buffer, lags=50, title=product, ax=axs[i])
  axs[i].grid()

ACFWeekResult = {
    "Men's Street Footwear": 1,
    "Women's Athletic Footwear": 0,
    "Women's Street Footwear": 1,
    "Women's Apparel": 1, 
    "Men's Athletic Footwear": 0,
    "Men's Apparel": 0
}

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(rawData["Product"].unique()):
  buffer = rawData[rawData["Product"] == product].copy()
  buffer["index"] =  pd.to_datetime(buffer["index"])
  buffer = buffer.sort_values(by="index").set_index("index").drop("Product", axis=1)
  buffer = buffer.resample("W", kind="timestamp").mean().diff().dropna()
  sm.graphics.tsa.plot_pacf(buffer, lags=25, title=product, ax=axs[i])
  axs[i].grid()

PACFWeekResult = {
    "Men's Street Footwear": 7,
    "Women's Athletic Footwear": 1,
    "Women's Street Footwear": 4,
    "Women's Apparel": 1, 
    "Men's Athletic Footwear": 8,
    "Men's Apparel": 8
}

bestModels = {}
treshold = "2021-10-24"
for product in rawData["Product"].unique():
  if len(bestModels) > 0:
    with open("drive/MyDrive/Adidas Sales/ARIMA_best_weekly_usual.pkl", "wb") as file:
      pkl.dump(bestModels, file)
  train, test = trainTestSplit(rawData, product, treshold, freq="W")
  bestParams = performEstiation(train, test, ACFWeekResult[product], 1, PACFWeekResult[product])
  if (product not in bestModels) or (bestParams["RMSE"] < bestModels[product]["RMSE"]):
    print("Product: {} (p,d,q): {} rmse: {}".format(product, bestParams["(p,d,q)"], bestParams["RMSE"]))
    bestModels[product] = bestParams

for product in bestModels:
  print("Product: {} (p,d,q): {} rmse: {}".format(product, bestModels[product]["(p,d,q)"], bestModels[product]["RMSE"]))