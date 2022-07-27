# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:28:27 2022

@author: Edwyn
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#%%
"""
Part 1 - Whole dataset
"""

#%%
keras.backend.clear_session()

#%%
file_path = r"C:\Users\edwyn\Documents\AI07\Datasets\Garment_Employees_Productivity\garments_worker_productivity.csv"
data = pd.read_csv(file_path)
df = data.copy()
df.shape

df.isna().sum()

df.fillna(0,inplace=True)

newdate = pd.to_datetime(df['date'],format='%m/%d/%Y')

date_df=pd.DataFrame({
                    "month": newdate.dt.month,
                    "day": newdate.dt.day,
                    "dayofweek": newdate.dt.weekday
                    })
df = df.drop(columns=['date','day'])
df = pd.concat([df,date_df],axis=1)

df['quarter'] = df['quarter'].str.replace("Quarter","").astype(int)
df['quarter'] = df['quarter'] - 1
df['team'] = df['team'] - 1

df['department'] = df['department'].replace("finishing ","finishing")
df['department'] = df['department'].replace("sweing","sewing")

df['dayofweek_sin'] = np.sin(df['dayofweek'] * (2 * np.pi / 7))
df['dayofweek_cos'] = np.cos(df['dayofweek'] * (2 * np.pi / 7))

df['quarter_sin'] = np.sin(df['quarter'] * (2 * np.pi / 5))
df['quarter_cos'] = np.cos(df['quarter'] * (2 * np.pi / 5))

df['day_sin'] = np.sin(df['day'] * (2 * np.pi / 31))
df['day_cos'] = np.cos(df['day'] * (2 * np.pi / 31))

df_sewing = df[df['department']=='sewing']
df_fin = df[df['department'] == 'finishing']

df_sewing = pd.get_dummies(df_sewing,columns=['team','month'],drop_first=True)
df_sewing = df_sewing.drop(columns=['department','day'])

df = pd.get_dummies(df,columns=['team','month','department'],drop_first=True)
df = df.drop(columns=['day'])

#%%
features = df.copy()
labels = features.pop('actual_productivity')


#%%
SEED = 54321
x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size=0.2,random_state=SEED)

#%%
standardizer = StandardScaler()
x_train_std = standardizer.fit_transform(x_train)
x_test_std = standardizer.transform(x_test)

#%%
nIn = x_train.shape[1]

inputs = keras.Input(shape=(nIn,))

h1 = layers.Dense(128, activation='elu')
h2 = layers.Dense(64, activation='elu')
h3 = layers.Dense(32, activation='elu')

out_layer = layers.Dense(1)

#%%
x = h1(inputs)
x = h2(x)
x = h3(x)
x = layers.Dropout(0.1)(x)

outputs = out_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary

#%%
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#%%
es = EarlyStopping(patience=15, verbose=1, restore_best_weights=True)

#%%
EPOCHS = 100
BATCH_SIZE = 64
history = model.fit(x_train_std, y_train, validation_split = 0.25, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es])

#%%
training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['mae']
val_acc = history.history['val_mae']
epochs_x_axis=history.epoch

plt.plot(epochs_x_axis, training_loss, label = 'Training Loss')
plt.plot(epochs_x_axis, val_loss, label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis, training_acc, label = 'Training Loss')
plt.plot(epochs_x_axis, val_acc, label='Validation Loss')
plt.title("Training vs Validation MAE")
plt.legend()
plt.figure()
plt.show()

#%%
predictions = model.predict(x_test_std)

pred_vs_label = np.concatenate((predictions, np.expand_dims(y_test, axis=1)),axis=1)

#%%
plt.figure(figsize=(10,10))
plt.scatter(y_test, predictions, c='b')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Prediction vs Actual')
plt.show()

#%%
print(model.evaluate(x_test_std,y_test))

#%%
"""
Part 2 - 'sewing' department only
"""

#%%
keras.backend.clear_session()

#%%
features_sew = df_sewing.copy()
labels_sew = features_sew.pop('actual_productivity')

#%%
SEED = 54321
x_train_sew, x_test_sew, y_train_sew, y_test_sew = train_test_split(features_sew, labels_sew, test_size=0.2, random_state=SEED)

#%%
standardizer = StandardScaler()
x_train_sew = standardizer.fit_transform(x_train_sew)
x_test_sew = standardizer.transform(x_test_sew)

#%%
nIn = x_train_sew.shape[1]

inputs = keras.Input(shape=(nIn,))

h1 = layers.Dense(128, activation='elu')
h2 = layers.Dense(64, activation='elu')
h3 = layers.Dense(32, activation='elu')

out_layer = layers.Dense(1)

#%%
x = h1(inputs)
x = h2(x)
x = h3(x)
x = layers.Dropout(0.1)(x)

outputs = out_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary

#%%
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#%%
es = EarlyStopping(patience=15, verbose=1, restore_best_weights=True)

#%%
EPOCHS = 100
BATCH_SIZE = 64
history = model.fit(x_train_sew, y_train_sew, validation_split = 0.25, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es])

#%%
training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['mae']
val_acc = history.history['val_mae']
epochs_x_axis=history.epoch

plt.plot(epochs_x_axis, training_loss, label = 'Training Loss')
plt.plot(epochs_x_axis, val_loss, label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis, training_acc, label = 'Training Loss')
plt.plot(epochs_x_axis, val_acc, label='Validation Loss')
plt.title("Training vs Validation MAE")
plt.legend()
plt.figure()
plt.show()

#%%
predictions = model.predict(x_test_sew)

pred_vs_label = np.concatenate((predictions, np.expand_dims(y_test_sew, axis=1)),axis=1)

#%%
plt.figure(figsize=(10,10))
plt.scatter(y_test_sew, predictions, c='b')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Prediction vs Actual - "sewing" department')
plt.show()

#%%
print(model.evaluate(x_test_sew,y_test_sew))

#%%
"""
Part 3 - 'finishing' department only
"""

#%%
keras.backend.clear_session()

#%%
df_fin = df_fin.drop(columns=['department','wip', 'idle_time', 'idle_men','no_of_style_change','incentive','day'])
df_fin = pd.get_dummies(df_fin,columns=['team','month'],drop_first=True)

#%%
features_fin = df_fin.copy()
labels_fin = features_fin.pop('actual_productivity')

#%%
SEED = 54321
x_train_fin, x_test_fin, y_train_fin, y_test_fin = train_test_split(features_fin, labels_fin, test_size=0.2, random_state=SEED)

#%%
standardizer = StandardScaler()
x_train_fin = standardizer.fit_transform(x_train_fin)
x_test_fin = standardizer.transform(x_test_fin)

#%%
nIn = x_train_fin.shape[1]

inputs = keras.Input(shape=(nIn,))

h1 = layers.Dense(128, activation='elu')
h2 = layers.Dense(64, activation='elu')
h3 = layers.Dense(32, activation='elu')

out_layer = layers.Dense(1)

#%%
x = h1(inputs)
x = h2(x)
x = h3(x)
x = layers.Dropout(0.1)(x)

outputs = out_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary
#%%
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#%%
es = EarlyStopping(patience=15, verbose=1, restore_best_weights=True)

#%%
EPOCHS = 100
BATCH_SIZE = 64
history2= model.fit(x_train_fin, y_train_fin, validation_split = 0.25, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es])

#%%
training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['mae']
val_acc = history.history['val_mae']
epochs_x_axis=history.epoch

plt.plot(epochs_x_axis, training_loss, label = 'Training Loss')
plt.plot(epochs_x_axis, val_loss, label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis, training_acc, label = 'Training Loss')
plt.plot(epochs_x_axis, val_acc, label='Validation Loss')
plt.title("Training vs Validation MAE")
plt.legend()
plt.figure()
plt.show()

#%%
predictions = model.predict(x_test_fin)

pred_vs_label = np.concatenate((predictions, np.expand_dims(y_test_fin, axis=1)),axis=1)

#%%
plt.figure(figsize=(10,10))
plt.scatter(y_test_fin, predictions, c='b')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Prediction vs Actual - "finishing" department ')
plt.show()

#%%
print(model.evaluate(x_test_fin,y_test_fin))
