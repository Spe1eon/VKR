import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
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

dailyTreshold = "2021-11-01"
weeklyTreshold = "2021-10-24"

dailyData_train, dailyData_test = dailyData.loc[dailyData["index"] <= pd.Timestamp(dailyTreshold)].copy(), dailyData.loc[dailyData["index"] > pd.Timestamp(dailyTreshold)].copy()

weeklyData_train, weeklyData_test = weeklyData.loc[weeklyData["index"] <= pd.Timestamp(weeklyTreshold)].copy(), weeklyData.loc[weeklyData["index"] > pd.Timestamp(weeklyTreshold)].copy()

#Согласно https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ru данные перед подачей в нейронную сеть надо нормализовать

dailyStats, weeklyStats = {}, {}
dailyData_train_clear, weeklyData_train_clear = pd.DataFrame(), pd.DataFrame()
dailyData_test_clear, weeklyData_test_clear = pd.DataFrame(), pd.DataFrame()
for product in dailyData_train["Product"].unique():
  bufferDaily = dailyData_train[dailyData_train["Product"] == product].copy()
  bufferWeekly = weeklyData_train[weeklyData_train["Product"] == product].copy()

  dailyStats[product], weeklyStats[product] = {}, {}

  dailyStats[product]["mean"], dailyStats[product]["std"] = bufferDaily["Units Sold"].mean(), bufferDaily["Units Sold"].std()
  weeklyStats[product]["mean"], weeklyStats[product]["std"] = bufferWeekly["Units Sold"].mean(), bufferWeekly["Units Sold"].std()

  bufferDaily["Units Sold"] = (bufferDaily["Units Sold"] - dailyStats[product]["mean"])/dailyStats[product]["std"]
  bufferWeekly["Units Sold"] = (bufferWeekly["Units Sold"] - weeklyStats[product]["mean"])/weeklyStats[product]["std"]

  if len(dailyData_train_clear) == 0:
    dailyData_train_clear = bufferDaily
    weeklyData_train_clear = bufferWeekly
  else:
    dailyData_train_clear = pd.concat([dailyData_train_clear, bufferDaily])
    weeklyData_train_clear = pd.concat([weeklyData_train_clear, bufferWeekly])
  
  bufferDaily = dailyData_test[dailyData_test["Product"] == product].copy()
  bufferWeekly = weeklyData_test[weeklyData_test["Product"] == product].copy()

  bufferDaily["Units Sold"] = (bufferDaily["Units Sold"] - dailyStats[product]["mean"])/dailyStats[product]["std"]
  bufferWeekly["Units Sold"] = (bufferWeekly["Units Sold"] - weeklyStats[product]["mean"])/weeklyStats[product]["std"]

  if len(dailyData_train_clear) == 0:
    dailyData_test_clear = bufferDaily
    weeklyData_test_clear = bufferWeekly
  else:
    dailyData_test_clear = pd.concat([dailyData_test_clear, bufferDaily])
    weeklyData_test_clear = pd.concat([weeklyData_test_clear, bufferWeekly])

#По аналогии с https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ru будем обучать модели на прогнозирование значений на основе окна последовательных выборок из данных.

class WindowGenerator():
  def __init__(self, input_width, label_width, shift, train_df=None, test_df=None, label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32)

    ds = ds.map(self.split_window)

    return ds
  
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

  history = model.fit(window.train, epochs=MAX_EPOCHS, callbacks=[early_stopping], verbose=0)
  return history
  
def cnn_model(num_filters, label_width, conv_width, num_features, activation):
  
  model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, conv_width, features]
    tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(num_filters, activation=activation, kernel_size=(conv_width)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(label_width*num_features, kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([label_width, num_features])
  ])

  return model

#Для начала напишем CNN сети для одномерного случая

dailyTrainBuffer, dailyTestBuffer = pd.DataFrame(), pd.DataFrame()

# Сверточная нейронная сеть
def objectiveDaily1DConv(trial):
  #Горизонт прогнозирования в окне - 3 недели
  label_width = 21
  conv_width = trial.suggest_int('conv_width', 3, label_width)
  # Длинна тестового датасета 60: окно не должно быть длинее!!!
  input_width = label_width + conv_width - 3
  num_filters = trial.suggest_int('num_filters', input_width, 3*input_width)

  conv_model = cnn_model(num_filters, label_width, conv_width, 1, trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'gelu', 'sigmoid', 'tanh']))

  multi_window = WindowGenerator(input_width=input_width,
                               label_width=label_width,
                               shift=label_width, 
                               train_df=dailyTrainBuffer, 
                               test_df=dailyTestBuffer, 
                               label_columns=["Units Sold"])

  history = compile_and_fit(conv_model, multi_window)

  loss = conv_model.evaluate(multi_window.test, verbose=0, return_dict=True)

  return math.sqrt(conv_model.evaluate(multi_window.test, verbose=0, return_dict=True)["loss"])

daily1DConvModels = {}
train_time = 3600
for product in dailyData_train_clear["Product"].unique():
  dailyTrainBuffer = dailyData_train_clear[dailyData_train_clear["Product"] == product].drop(["index", "Product"], axis=1)
  dailyTestBuffer = dailyData_test_clear[dailyData_test_clear["Product"] == product].drop(["index", "Product"], axis=1)
  study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='1D_CNN_Daily')
  study.optimize(objectiveDaily1DConv, timeout=train_time, show_progress_bar=True)
  trial = study.best_trial
  daily1DConvModels[product] = trial.params

pd.DataFrame(daily1DConvModels, index=daily1DConvModels[product].keys())

for product in daily1DConvModels.keys():
  label_width = 21
  conv_width = daily1DConvModels[product]["conv_width"]
  input_width = label_width + conv_width - 3
  num_filters = daily1DConvModels[product]["num_filters"]
  activation = daily1DConvModels[product]["activation"]
  
  conv_model = cnn_model(num_filters, label_width, conv_width, 1, activation)

  dailyTrainBuffer = dailyData_train_clear[dailyData_train_clear["Product"] == product].drop(["index", "Product"], axis=1)
  dailyTestBuffer = dailyData_test_clear[dailyData_test_clear["Product"] == product].drop(["index", "Product"], axis=1)

  multi_window = WindowGenerator(input_width=input_width,
                               label_width=label_width,
                               shift=label_width, 
                               train_df=dailyTrainBuffer, 
                               test_df=dailyTestBuffer, 
                               label_columns=["Units Sold"])

  history = compile_and_fit(conv_model, multi_window)

  loss = math.sqrt(conv_model.evaluate(multi_window.test, verbose=0, return_dict=True)["loss"])

  print("Product: {} rmse: {}".format(product, loss))

weeklyTrainBuffer, weeklyTestBuffer = pd.DataFrame(), pd.DataFrame()

def objectiveWeekly1DConv(trial):
  #Горизонт прогнозирования в окне - 3 недели
  label_width = 3
  conv_width = trial.suggest_int('conv_width', 3, label_width)
  # Длинна тестового датасета 10: окно не должно быть длинее!!!
  input_width = label_width + conv_width - 1
  num_filters = trial.suggest_int('num_filters', input_width, 3*input_width)

  multi_conv_model = cnn_model(num_filters, label_width, conv_width, 1, trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'gelu', 'sigmoid', 'tanh']))

  multi_window = WindowGenerator(input_width=input_width,
                               label_width=label_width,
                               shift=label_width, 
                               train_df=weeklyTrainBuffer, 
                               test_df=weeklyTestBuffer, 
                               label_columns=["Units Sold"])

  history = compile_and_fit(multi_conv_model, multi_window)

  return math.sqrt(multi_conv_model.evaluate(multi_window.test, verbose=0, return_dict=True)["loss"])

weekly1DConvModels = {}
train_time = 3600
for product in dailyData_train_clear["Product"].unique():
  weeklyTrainBuffer = weeklyData_train_clear[weeklyData_train_clear["Product"] == product].drop(["index", "Product"], axis=1)
  weeklyTestBuffer = weeklyData_test_clear[weeklyData_test_clear["Product"] == product].drop(["index", "Product"], axis=1)
  study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='1D_CNN_Weekly')
  study.optimize(objectiveWeekly1DConv, timeout=train_time, show_progress_bar=True)
  trial = study.best_trial
  weekly1DConvModels[product] = trial.params

pd.DataFrame(weekly1DConvModels, index=weekly1DConvModels[product].keys())

for product in weekly1DConvModels.keys():
  label_width = 3
  conv_width = weekly1DConvModels[product]["conv_width"]
  input_width = label_width + conv_width - 1
  num_filters = weekly1DConvModels[product]["num_filters"]
  activation = weekly1DConvModels[product]["activation"]
  
  conv_model = cnn_model(num_filters, label_width, conv_width, 1, activation)

  weeklyTrainBuffer = weeklyData_train_clear[weeklyData_train_clear["Product"] == product].drop(["index", "Product"], axis=1)
  weeklyTestBuffer = weeklyData_test_clear[weeklyData_test_clear["Product"] == product].drop(["index", "Product"], axis=1)

  multi_window = WindowGenerator(input_width=input_width,
                               label_width=label_width,
                               shift=label_width, 
                               train_df=weeklyTrainBuffer, 
                               test_df=weeklyTestBuffer, 
                               label_columns=["Units Sold"])

  history = compile_and_fit(conv_model, multi_window)

  loss = math.sqrt(conv_model.evaluate(multi_window.test, verbose=0, return_dict=True)["loss"])

  print("Product: {} rmse: {}".format(product, loss))

#Теперь приступаем к обучению LSTM сетей для обномерного случая

def lstm_model(num_units, label_width, num_features, activation, rec_activation):
  
  model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(num_units, return_sequences=False, activation=activation, recurrent_activation=rec_activation),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(label_width*num_features, kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([label_width, num_features])
])

  return model

# Реккурентная нейронная сеть
def objectiveDaily1DLSTM(trial):
  #Горизонт прогнозирования в окне - 3 недели
  label_width = 21
  num_units = trial.suggest_int('num_units', label_width, 3*label_width)

  activation = trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'gelu', 'sigmoid', 'tanh'])
  rec_activation = trial.suggest_categorical('rec_activation', ['relu', 'elu', 'selu', 'gelu', 'sigmoid', 'tanh'])

  model = lstm_model(num_units, label_width, 1, activation, rec_activation)

  window = WindowGenerator(input_width=label_width,
                               label_width=label_width,
                               shift=label_width, 
                               train_df=dailyTrainBuffer, 
                               test_df=dailyTestBuffer, 
                               label_columns=["Units Sold"])

  history = compile_and_fit(model, window)

  loss = model.evaluate(window.test, verbose=0, return_dict=True)

  return math.sqrt(model.evaluate(window.test, verbose=0, return_dict=True)["loss"])

daily1DLSTMModels = {}
train_time = 3600
for product in dailyData_train_clear["Product"].unique():
  dailyTrainBuffer = dailyData_train_clear[dailyData_train_clear["Product"] == product].drop(["index", "Product"], axis=1)
  dailyTestBuffer = dailyData_test_clear[dailyData_test_clear["Product"] == product].drop(["index", "Product"], axis=1)
  study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='1D_LSTM_Daily')
  study.optimize(objectiveDaily1DLSTM, timeout=train_time, show_progress_bar=True)
  trial = study.best_trial
  daily1DLSTMModels[product] = trial.params

pd.DataFrame(daily1DLSTMModels, index=daily1DLSTMModels[product].keys())

for product in daily1DLSTMModels.keys():
  label_width = 21
  num_units = daily1DLSTMModels[product]["num_units"]
  activation = daily1DLSTMModels[product]["activation"]
  rec_activation = daily1DLSTMModels[product]["rec_activation"]

  dailyTrainBuffer = dailyData_train_clear[dailyData_train_clear["Product"] == product].drop(["index", "Product"], axis=1)
  dailyTestBuffer = dailyData_test_clear[dailyData_test_clear["Product"] == product].drop(["index", "Product"], axis=1)

  model = lstm_model(num_units, label_width, 1, activation, rec_activation)

  window = WindowGenerator(input_width=input_width,
                               label_width=label_width,
                               shift=label_width, 
                               train_df=dailyTrainBuffer, 
                               test_df=dailyTestBuffer, 
                               label_columns=["Units Sold"])

  history = compile_and_fit(model, window)

  loss = math.sqrt(model.evaluate(window.test, verbose=0, return_dict=True)["loss"])

  print("Product: {} rmse: {}".format(product, loss))

def objectiveWeekly1DLSTM(trial):
  #Горизонт прогнозирования в окне - 3 недели
  label_width = 3
  num_units = trial.suggest_int('num_units', label_width, 3*label_width)

  activation = trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'gelu', 'sigmoid', 'tanh'])
  rec_activation = trial.suggest_categorical('rec_activation', ['relu', 'elu', 'selu', 'gelu', 'sigmoid', 'tanh'])

  model = lstm_model(num_units, label_width, 1, activation, rec_activation)

  window = WindowGenerator(input_width=label_width,
                               label_width=label_width,
                               shift=label_width, 
                               train_df=weeklyTrainBuffer, 
                               test_df=weeklyTestBuffer, 
                               label_columns=["Units Sold"])

  history = compile_and_fit(model, window)

  return math.sqrt(model.evaluate(window.test, verbose=0, return_dict=True)["loss"])

weekly1DLSTMModels = {}
train_time = 3600
for product in dailyData_train_clear["Product"].unique():
  weeklyTrainBuffer = weeklyData_train_clear[weeklyData_train_clear["Product"] == product].drop(["index", "Product"], axis=1)
  weeklyTestBuffer = weeklyData_test_clear[weeklyData_test_clear["Product"] == product].drop(["index", "Product"], axis=1)
  study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='1D_LSTM_Weekly')
  study.optimize(objectiveDaily1DLSTM, timeout=train_time, show_progress_bar=True)
  trial = study.best_trial
  weekly1DLSTMModels[product] = trial.params

pd.DataFrame(weekly1DLSTMModels, index=weekly1DLSTMModels[product].keys())

for product in weekly1DLSTMModels.keys():
  label_width = 3
  num_units = weekly1DLSTMModels[product]["num_units"]
  activation = weekly1DLSTMModels[product]["activation"]
  rec_activation = weekly1DLSTMModels[product]["rec_activation"]

  weeklyTrainBuffer = weeklyData_train_clear[weeklyData_train_clear["Product"] == product].drop(["index", "Product"], axis=1)
  weeklyTestBuffer = weeklyData_test_clear[weeklyData_test_clear["Product"] == product].drop(["index", "Product"], axis=1)

  model = lstm_model(num_units, label_width, 1, activation, rec_activation)

  window = WindowGenerator(input_width=input_width,
                               label_width=label_width,
                               shift=label_width, 
                               train_df=weeklyTrainBuffer, 
                               test_df=weeklyTestBuffer, 
                               label_columns=["Units Sold"])

  history = compile_and_fit(model, window)

  loss = math.sqrt(model.evaluate(window.test, verbose=0, return_dict=True)["loss"])

  print("Product: {} rmse: {}".format(product, loss))

#Ссылки:
#1. https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ru