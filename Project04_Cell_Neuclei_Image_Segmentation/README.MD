# Object Segmentation of Cell Neuclei using Deep Learning Neural Network
## 1. Overview
#### The purpose of this project is to accurately predicting the segments of cell neuclei in images. This project is created using Spyder IDE. This model is trained using [Data Science Bowl 2018 Dataset](https://www.kaggle.com/competitions/data-science-bowl-2018/overview)

## 2. Methodology
### 2.1 Preprocessing
#### a. The dataset is already split into 'train' and 'test' folders each containing images as well as masks of their respective images.
#### b. The data is loaded and training data is split into training:validation with ratio of 80:20.

#### Some sample images with their respecting masking
![image](https://user-images.githubusercontent.com/82880708/182068583-7b0d6174-2016-40e0-8b61-86696cdb5aa3.png)
![image](https://user-images.githubusercontent.com/82880708/182068590-917f0daf-9980-47ae-a98e-814b2972f6ec.png)
![image](https://user-images.githubusercontent.com/82880708/182068652-0c91fc27-e337-46dc-91fc-9060fc6f3c03.png)
![image](https://user-images.githubusercontent.com/82880708/182068667-8de93e31-ea97-42be-b625-4d961267b136.png)
#### Prediction sample before model training

![image](https://user-images.githubusercontent.com/82880708/182069772-02c2d668-dddc-401e-81a0-482fed4f5605.png)


### 2.2 Model Pipeling & Training
#### a. Data augmentation RandomFlip is applied on the training dataset.
#### b. MobileNetV2 is used as transfer learning for the base model.
#### c. U-Net is used as the model architecture which serve as the feature extractor and upward stack.

| Layer (type) | Output Shape | Param # | Connected to |
| --- | --- | --- | --- |
| input_2 (InputLayer) | [(None, 128, 128, 3)] | 0 | [] |
| model (Functional) | [(None, 64, 64, 96),(None, 32, 32, 144),(None, 16, 16, 192),(None, 8, 8, 576),(None, 4, 4, 320)] | 1841984 | ['input_2[0][0]'] |  
| sequential (Sequential) | (None, 8, 8, 512) | 1476608 | ['model[0][4]'] |
| concatenate (Concatenate) | (None, 8, 8, 1088) | 0 | ['sequential[0][0]','model[0][3]'] |
| sequential_1 (Sequential) | (None, 16, 16, 256) | 2507776 | ['concatenate[0][0]'] |
| concatenate_1 (Concatenate) | (None, 16, 16, 448)  | 0 | ['sequential_1[0][0]','model[0][2]'] |
| sequential_2 (Sequential) | (None, 32, 32, 128) | 516608 | ['concatenate[0][0]'] |
| concatenate_2 (Concatenate) | (None, 32, 32, 272) | 0 | ['sequential_2[0][0]','model[0][1]'] |
| sequential_3 (Sequential) | (None, 64, 64, 64) | 156928 | ['concatenate_2[0][0]'] |
| concatenate_3 (Concatenate) | (None, 64, 64, 160) | 0 | ['sequential_3[0][0]','model[0][0]'] |
| conv2d_transpose_4 (Conv2DTranspose) | (None, 128, 128, 2) | 2882 | ['concatenate_3[0][0]'] |

#### Total params: 6,502,786
#### Trainable params: 4,658,882
#### Non-trainable params: 1,843,904
#### This model is trained with 100 epochs with EarlyStopping patience of 20.

### 3. Results
#### EarlyStopping is applied at epoch 70 with best weights at epoch 50.

![image](https://user-images.githubusercontent.com/82880708/182070732-716ed918-ed9f-438a-add7-b0094885a6a9.png)
![image](https://user-images.githubusercontent.com/82880708/182070736-276697c1-4bf5-4934-bfaf-ac35f0c9a5b1.png)

#### Validation Loss: 0.0971
#### Validation Accuracy: 0.9635

#### Testing Loss: 0.089142
#### Testing Accuracy: 0.963795

#### Here are some testing predictions compared with actual.
![image](https://user-images.githubusercontent.com/82880708/182070805-dfc89e6b-72df-4e1a-91af-6aae5f616693.png)
![image](https://user-images.githubusercontent.com/82880708/182070812-ea39f467-a874-4eaa-98d5-57bf741f8f69.png)
![image](https://user-images.githubusercontent.com/82880708/182070817-4d278fc0-4446-48ef-9020-0f687641e784.png)

This model is capable of image segmentation of cell neuclei with high accuracy.
