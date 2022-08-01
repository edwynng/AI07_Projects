# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:53:07 2022

@author: Edwyn
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, callbacks
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2

#%%
root_path = r"C:\Users\edwyn\Documents\AI07\Datasets\Cell_Neuclei"
train_path = os.path.join(root_path,'train')
input_path = os.path.join(train_path,'inputs')
mask_path = os.path.join(train_path,'masks')

test_path = os.path.join(root_path,'test')
test_image_path = os.path.join(test_path,'inputs')
test_mask_path = os.path.join(test_path,'masks')
    
#%%
IMG_SIZE = (128,128)

def normalize(input_image,input_mask):
    input_image = tf.cast(input_image,tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image(path):
    images = []
    for image in os.listdir(path):
        file = cv2.imread(os.path.join(path,image))
        file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
        file = cv2.resize(file,IMG_SIZE)
        images.append(file)
    return images

def load_mask(path):
    masks = []    
    for mask in os.listdir(path):
        file = cv2.imread(os.path.join(path,mask),cv2.IMREAD_GRAYSCALE)
        file = cv2.resize(file,IMG_SIZE)
        masks.append(file)
    return masks
        

train_images = load_image(input_path)
train_masks = load_mask(mask_path)
test_images = load_image(test_image_path)
test_masks = load_mask(test_mask_path)

#%%
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = train_images[i]
    plt.imshow(img_plot)
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(train_masks[i])
    plt.axis('off')

plt.show()

#%%    
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)
print(train_masks_np_exp[0].min(),train_masks_np_exp[0].max())

#%%
train_converted_masks = np.ceil(train_masks_np_exp/255)
test_converted_masks = np.ceil(test_masks_np_exp/255)
train_converted_masks = 1 - train_converted_masks
test_converted_masks = 1 - test_converted_masks

print(np.unique(train_converted_masks[0]))

#%%
train_converted_images = train_images_np / 255.0
test_converted_images = test_images_np / 255.0

#%%
SEED = 54321
x_train, x_val, y_train, y_val = train_test_split(train_converted_images, train_converted_masks, test_size=0.2,random_state=SEED)

#%%
x_train_tensor = tf.data.Dataset.from_tensor_slices(x_train)
x_val_tensor = tf.data.Dataset.from_tensor_slices(x_val)
x_test_tensor = tf.data.Dataset.from_tensor_slices(test_converted_images)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_val_tensor = tf.data.Dataset.from_tensor_slices(y_val)
y_test_tensor = tf.data.Dataset.from_tensor_slices(test_converted_masks)

#%%
train_dataset = tf.data.Dataset.zip((x_train_tensor,y_train_tensor))
val_dataset = tf.data.Dataset.zip((x_val_tensor,y_val_tensor))
test_dataset = tf.data.Dataset.zip((x_test_tensor,y_test_tensor))

#%%
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

#%%
train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

val_batches = (
    val_dataset
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

test_batches = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

#%%
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
#%%
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
    
#%%
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False)

layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack  = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=IMG_SHAPE)
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack,skips):
        x=up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    last = layers.Conv2DTranspose(
        filters=output_channels,kernel_size=3,strides=2,padding='same')
    #64x64 ---> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs,outputs=x)

#%%
OUTPUT_CLASSES = len(np.unique(train_converted_masks[0]))

model = unet_model(output_channels=OUTPUT_CLASSES)
model.summary()
#%%
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
keras.utils.plot_model(model, show_shapes=True)

#%%

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)[0]])
            
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))[0]])
        
#%%
show_predictions()

#%%
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))

es = callbacks.EarlyStopping(patience=20,verbose=1,restore_best_weights=True)
#%%
EPOCHS = 100
VALIDATION_STEPS = len(val_dataset)//BATCH_SIZE
history = model.fit(train_batches,validation_data=val_batches,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,callbacks=[DisplayCallback(),es])

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
show_predictions(test_batches,3)

#%%
print(model.evaluate(test_batches))
