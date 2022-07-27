# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:06:13 2022

@author: Edwyn
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#%%
file_path = r"C:\Users\edwyn\Documents\AI07\Datasets\Heart_Disease\heart.csv"
file = pd.read_csv(file_path)

print(file.head(5))
print(file.isna().sum())

#%%
features = file.copy()
labels = features.pop('target')

#%%
SEED = 54321
x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size=0.2,random_state=SEED)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#%%
standardizer = StandardScaler()
x_train_std = standardizer.fit_transform(x_train)
x_test_std = standardizer.transform(x_test)

#%%
nIn = features.shape[1]
nClass = len(np.unique(labels))

inputs = layers.Input(shape=(nIn,))

h1 = layers.Dense(128,activation='relu')
h2 = layers.Dense(64,activation='relu')
h3 = layers.Dense(32,activation='relu')


out_layer = layers.Dense(nClass,activation='softmax')

#%%
x = h1(inputs)
x = h2(x)
x = h3(x)

outputs = out_layer(x)
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#%%
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%%
EPOCHS = 20
BATCH_SIZE=32
history = model.fit(x_train_std,y_train,validation_split=0.2,batch_size=BATCH_SIZE,epochs=EPOCHS)

#%%
training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_x_axis=history.epoch

plt.plot(epochs_x_axis, training_loss, label = 'Training Loss')
plt.plot(epochs_x_axis, val_loss, label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis, training_acc, label = "Training Accuracy")
plt.plot(epochs_x_axis, val_acc, label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.figure()

plt.show()
#%%
predictions = np.argmax(model.predict(x_test_std),axis=1)
actual_vs_predictions = np.transpose(np.vstack((y_test,predictions)))

print(model.evaluate(x_test_std,y_test))

#%%
print(confusion_matrix(y_test, predictions))