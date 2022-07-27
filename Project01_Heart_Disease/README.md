# **Predicting Heart Disease Using Feedforward Neural Network**
## **1. Overview**
#### The purpose of this project is to create an accurate deep learning model to predict the presence of heart disease for patients. This project is created using Spyder IDE. This model is trained using:
[Link to Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

## **2. Methodology**
### 2.1 Preprocessing
#### a. The data is loaded and the dataset is already complete.
#### b. The data is split into train-test set with a ratio of 80:20.
#### c. The features are normalized using standardization.
#### d. Each dense layer uses 'relu' as activation function.

### 2.2 Model Pipeline & Evaluation

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

![image](https://user-images.githubusercontent.com/82880708/180780568-3232e302-7300-4477-84f6-421261d7e5be.png)
![image](https://user-images.githubusercontent.com/82880708/180780586-4df02159-a186-4796-846b-8fbf06e8a076.png)

||Predicted Positive|Predicted Negative|
|---|---|---|
|Actual Positive|91|1|
|Actual Negative|1|112|

#### This model is trained on epochs of 20 and batch size of 32 with validation split of 20%. The training accuracy is 99.85% and validation accuracy is 97.56%. This may be a sign of overfitting, however the testing accuracy is 99.02%.
