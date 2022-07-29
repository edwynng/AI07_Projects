# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:26:42 2022

@author: Edwyn
"""

import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

#%%
root_path = r"C:\Users\edwyn\Documents\AI07\Datasets\Concrete_Cracks"

data_dir = pathlib.Path(root_path)

BATCH_SIZE = 32
IMG_SIZE = (227,227)
SEED = 54321

train_data = keras.utils.image_dataset_from_directory(data_dir,shuffle=True, image_size=IMG_SIZE, validation_split=0.4, subset='training',seed=SEED)
val_data = keras.utils.image_dataset_from_directory(data_dir,shuffle=True, image_size=IMG_SIZE, validation_split=0.4, subset='validation',seed=SEED)

#%%
val_batches = tf.data.experimental.cardinality(val_data)

test_data = val_data.take(val_batches//2)
val_data = val_data.skip(val_batches//2)

#%%
class_names = train_data.class_names
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_data.prefetch(buffer_size=AUTOTUNE)
pf_val = val_data.prefetch(buffer_size=AUTOTUNE)
pf_test = test_data.prefetch(buffer_size=AUTOTUNE)

#%%
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
preprocess_input = applications.mobilenet_v2.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)

base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False, weights='imagenet')

base_model.trainable = True

for layer in base_model.layers[:150]:
    layer.trainable = False
    
base_model.summary()

#%%
global_avg = layers.GlobalAveragePooling2D()

out_layer = layers.Dense(len(class_names),activation = 'softmax')

#%%
inputs = keras.Input(shape=IMG_SHAPE)

x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x)
x = global_avg(x)
x = layers.Dropout(0.3)(x)

outputs = out_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#%%
lr_schedule = optimizers.schedules.CosineDecay(0.001, 500)

optimizer = optimizers.Adam(learning_rate = lr_schedule)
loss = losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

es = EarlyStopping(patience=5,verbose=1,restore_best_weights=True)
#%%
EPOCHS = 10
history = model.fit(pf_train, validation_data=pf_val, epochs=EPOCHS, callbacks=[es])

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
test_loss, test_accuracy = model.evaluate(pf_test)

print("---------------------After Training----------------------")
print("Loss = ", test_loss)
print('Accuracy = ', test_accuracy)

#%%
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch),axis=1)

#%%
label_vs_prediction = np.transpose(np.vstack((label_batch,predictions)))

#%%
plt.figure(figsize=(15,15))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis("off")