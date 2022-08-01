# Detecting Concrete Cracks using Convolutional Neural Network Image Classification.

## 1. Overview
#### The purpose of this project is to create an accurate deep learning model to identify cracks on concretes. This project is created using Spyder IDE. This model is trained using: [Concrete crack images dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## 2. Methodology
### 2.1 Preprocessing
#### The data is already labeled in their respective folders (Positive, Negative). The data is loaded and split into train:validation:test with ratio of 60:20:20.

### 2.2 Model Pipeline & Training
#### a. Data augmentation RandomFlip and RandomRotatuin is applied on the training dataset.
#### b. MobileNetV2 is used as transfer learning for the base model and the first 150 layers is frozen.
#### c. GlobalAveragePooling2D layer is used for pooling.
#### d. The output layer is a Dense layer with 'softmax' activation

| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| input_2 (InputLayer) | [(None, 227, 227, 3)] | 0 |                                                                
| sequential (Sequential) | (None, 227, 227, 3) | 0 |   
| tf.math.truediv (TFOpLambda) | (None, 227, 227, 3) | 0 |
| tf.math.subtract (TFOpLambda) | (None, 227, 227, 3) | 0 |
| mobilenetv2_1.00_224 (Functional) | (None, 8, 8, 1280) | 2257984 |
| global_average_pooling2d (GlobalAveragePooling2D) | (None, 1280)  | 0 |
| dropout (Dropout) | (None, 1280) | 0 |
| dense (Dense) | (None, 2) | 2562 |

#### Total params: 2,260,546
#### Trainable params: 415,362
#### Non-trainable params: 1,845,184
#### This model is trained with 10 epochs with EarlyStopping patience of 5.

## 3. Results
#### EarlyStopping is applied at epoch 9 with best epoch of 4.
#### Validation Loss: 0.0046
#### Validation Accuracy: 0.9994

![image](https://user-images.githubusercontent.com/82880708/181775610-ed2116a0-0ea3-4dcb-9373-e821ef2fc56f.png)
![image](https://user-images.githubusercontent.com/82880708/181775617-15099b60-2715-416e-a090-8cbb36ef3266.png)

#### Testing Loss: 0.007529
#### Testing Accuracy: 0.998375

![image](https://user-images.githubusercontent.com/82880708/181775659-8ce001d8-a062-42b8-9036-648b6d0cb5cd.png)
#### Here are some predictions of testing images.
