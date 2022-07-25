# **Predicting Heart Disease Using Feedforward Neural Network**
## **1. Overview**
#### The purpose of this project is to create an accurate deep learning model to predict the presence of heart disease for patients. This project is created using Spyder IDE. This model is trained using:
[Link to Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

## **3. Methodology**
### 3.1 Preprocessing
#### a. The data is loaded and the dataset is already complete.
#### b. The data is split into train-test set with a ratio of 80:20.
#### c. The features are normalized using standardization.

### 3.2 Model Pipeline & Evaluation

 | Layer (type) | Output Shape | Param # |
 | --- | --- | --- |
 | input_1 (InputLayer) | [(None, 13)] | 0 |                                                                
 | dense (Dense) | (None, 128) | 1792 |   
 | dense_1 (Dense) | (None, 64) | 8256 |
 | dense_2 (Dense) | (None, 32) | 2080 |
 | dense_3 (Dense) | (None, 2) | 66 |

#### Total params: 12,194
#### Trainable params: 12,194
#### Non-trainable params: 0

![image](https://user-images.githubusercontent.com/82880708/180745191-5190aade-7841-4f38-9982-0578091a03ed.png)

#### This model is trained on epochs of 20 and batch size of 32 with validation split of 20%. The training accuracy and validation accuracy is 100%. This may be a sign of overfitting, however the testing accuracy is 96.59%.
