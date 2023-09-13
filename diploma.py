import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

rawData = pd.read_csv("Adidas Sales/Adidas US Sales Datasets.csv", delimiter = ";")
rawData.head(25)

# Упрощенная схема: удаляем столбцы "Retailer", "Retailer ID", "Region", "State", "City", "Price per Unit", "Total Sales", "Operating Profit", "Operating Margin", "Sales Method"
rawData.drop(["Retailer", "Retailer ID", "Region", "State", "City", "Price per Unit", "Total Sales", "Operating Profit", "Operating Margin", "Sales Method"], axis=1, inplace=True)
rawData["Invoice Date"] = pd.to_datetime(rawData["Invoice Date"])
rawData["Units Sold"] = rawData["Units Sold"].str.replace(" ", "").astype("int32")
products = rawData["Product"].unique()
dateRange = pd.date_range(rawData["Invoice Date"].min(), rawData["Invoice Date"].max())
# Аггрегирование по дате  
aggregatedData = rawData.groupby(["Product", "Invoice Date"]).sum().reset_index()
filledData = pd.DataFrame()
for product in products:
  buffer = aggregatedData[aggregatedData["Product"] == product].copy()
  buffer.set_index("Invoice Date", inplace=True)
  # buffer = buffer.reindex(dateRange, fill_value=0).reset_index()
  buffer = buffer.reindex(dateRange).reset_index()
  buffer["Product"] = product
  if len(filledData) == 0:
    filledData = buffer
  else:
    filledData = pd.concat([filledData, buffer])
filledData.head(25)

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(filledData["Product"].unique()):
  buffer = filledData[filledData["Product"] == product].copy()
  buffer["index"] = pd.to_datetime(buffer["index"])
  axs[i].plot(buffer["index"], buffer["Units Sold"])
  axs[i].set_title(product)
  axs[i].grid()

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
cutData = pd.DataFrame()
for i, product in enumerate(filledData["Product"].unique()):
  buffer = filledData[(filledData["Product"] == product) & (filledData["index"] > "2021-01-01")].copy()
  buffer["index"] = pd.to_datetime(buffer["index"])
  buffer.sort_values(by="index", inplace=True)
  # Попробуем заполнение полиномиальным сплайном 3 порядка (отрицательные значения приравняем 0)
  buffer.loc[buffer["Units Sold"].isna(), "Units Sold"] = buffer["Units Sold"].interpolate(method="polynomial", axis=0, order=3)
  # Избавляемся отрезультата < 0 и оставшихся пропусков
  buffer.loc[buffer["Units Sold"] < 0, "Units Sold"] = np.nan
  roll = buffer["Units Sold"].rolling(10, center=True, min_periods=1).mean()
  buffer.loc[buffer["Units Sold"].isna(), "Units Sold"] = roll
  if len(cutData) == 0:
    cutData = buffer
  else:
    cutData = pd.concat([cutData, buffer])
  axs[i].plot(buffer["index"], buffer["Units Sold"])
  axs[i].set_title(product)
  axs[i].grid()

cutData.to_csv("Adidas Sales/sales_simplified.csv")

rawData = pd.read_csv("Adidas Sales/sales_simplified.csv")
rawData.drop("Unnamed: 0", axis=1, inplace=True)

from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm

for product in rawData["Product"].unique():
  buffer = rawData[rawData["Product"] == product].copy()
  buffer["index"] =  pd.to_datetime(buffer["index"])
  buffer.sort_values(by="index", inplace=True)
  test = adfuller(buffer["Units Sold"])
  print("Product: '{}'\n\t'adf':{}\n\t'p-value: {}'\n\t'critical-values: {}'".format(product, test[0], test[1], test[4]))
  if test[1] > 0.05:
    print("\tРяд '{}' не стационарен: есть единичные корни\n".format(product))
  else:
    print("\tРяд '{}' стационарен: единичных корней нет\n".format(product))

for product in rawData["Product"].unique():
  buffer = rawData[rawData["Product"] == product].copy()
  buffer["index"] =  pd.to_datetime(buffer["index"])
  buffer.sort_values(by="index", inplace=True)
  test = kpss(buffer["Units Sold"])
  print("Product: '{}'\n\t'adf':{}\n\t'p-value: {}'\n\t'critical-values: {}'".format(product, test[0], test[1], test[3]))
  if test[1] < 0.05:
    print("\tРяд '{}' не стационарен\n".format(product))
  else:
    print("\tРяд '{}' стационарен\n".format(product))

# Усреднение по неделям: тест Дикки - Фуллера
for product in rawData["Product"].unique():
  buffer = rawData[rawData["Product"] == product].copy()
  buffer["index"] =  pd.to_datetime(buffer["index"])
  buffer.sort_values(by="index", inplace=True)
  buffer = buffer.resample("W", kind="timestamp", on="index").mean()
  test = adfuller(buffer["Units Sold"])
  print("Product: '{}'\n\t'adf':{}\n\t'p-value: {}'\n\t'critical-values: {}'".format(product, test[0], test[1], test[4]))
  if test[1] > 0.05:
    print("\tРяд '{}' не стационарен: есть единичные корни\n".format(product))
  else:
    print("\tРяд '{}' стационарен: единичных корней нет\n".format(product))

# Усреднение по неделям: тест Квятковского — Филлипса — Шмидта — Шина
for product in rawData["Product"].unique():
  buffer = rawData[rawData["Product"] == product].copy()
  buffer["index"] =  pd.to_datetime(buffer["index"])
  buffer.sort_values(by="index", inplace=True)
  buffer = buffer.resample("W", kind="timestamp", on="index").mean()
  test = kpss(buffer["Units Sold"])
  print("Product: '{}'\n\t'adf':{}\n\t'p-value: {}'\n\t'critical-values: {}'".format(product, test[0], test[1], test[3]))
  if test[1] < 0.05:
    print("\tРяд '{}' не стационарен\n".format(product))
  else:
    print("\tРяд '{}' стационарен\n".format(product))

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(rawData["Product"].unique()):
  Y = rawData[rawData["Product"] == product]["Units Sold"]
  sm.graphics.tsa.plot_acf(Y, lags=200, title=product, ax=axs[i])
  axs[i].grid()

fig, axs = plt.subplots(6, 1, figsize=(40, 25))
for i, product in enumerate(rawData["Product"].unique()):
  # Т.к. видно, что тренд в данных постоянный, удалим его, чтобы было лучше видно сезонные компоненты
  Y = rawData[rawData["Product"] == product]["Units Sold"] - (rawData[rawData["Product"] == product]["Units Sold"]).mean()
  X_FFT = np.fft.fftfreq(len(Y))
  X_FFT = X_FFT[:len(X_FFT)//2]  
  Y_FFT = np.fft.fft(Y)
  Y_FFT = np.abs(Y_FFT[:len(Y_FFT)//2])
  axs[i].plot(X_FFT, Y_FFT)
  axs[i].set_title(product)
  axs[i].grid()