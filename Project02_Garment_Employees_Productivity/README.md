# **Predicting Productivity of Garment Employees using Feedforward Neural Network**

## **1. Overview**
#### The purpose of this project is to create an accurate deep learning model to predict the actual productivity of garment employees. This project is created using Spyder IDE. This dataset is split into three parts: complete dataset, 'sewing' department only, 'finishing' department only. This model is trained using:
[Link to Garment Employees Productivity Dataset](https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees)

## 2. Methodology

### 2.1 Preprocessing
a. The dataset is loaded and checked for any NaN values.

|Column names | NaN count|
|---|---|
|date|0|
|quarter|0|
|department|0|
|day|0|
|team|0|
|targeted_productivity|0|
|smv|0|
|wip|506|
|over_time|0|
|incentive|0|
|idle_time|0|
|idle_men|0|
|no_of_style_change|0|
|no_of_workers|0|
|actual_productivity|0|

b. Since all missing wip (Work in progress) values are from finishing department, it makes sense to fill these with 0.

c. The 'day' column is renamed as 'dayofweek' and converted into integers where Monday is 0, Sunday is 6. The 'date' is converted into month and day as this may contain crucial information. 'quarter' is also converted into integer.

d. 'department' typos such as "finishing " and "sweing" has been corrected.

e. 'dayofweek', 'quarter', 'day' has been converted to angular distance. 

f. Some columns of data such as 'wip', 'idle_time', 'idle_men', 'no_of_style_change' only applies to 'sewing' department. I have decided to split this dataset into 3 parts: complete dataset (df), 'sewing' department only (df_sewing), 'finishing' department only (df_fin).

g. All data set undergo some One-Hot-Encoding using get_dummies and dropped some redundant features.
- For df, encoded 'team', 'month', 'department'
- For df_sewing, encoded 'team, 'month, dropped 'department'
- For df_fin, encoded 'team', 'month' ,dropped 'department','wip', 'idle_time', 'idle_men','no_of_style_change'

### 2.2 Model Pipeline
a. All dataset are split using train-test-split with ratio of 80:20.

b. All features are standardized using StandardScaler.

c. The pipeline is the same for all dataset.

Pipeline for df:
| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| input_1 (InputLayer) | [(None, 30)] | 0 |                                                                
| dense (Dense) | (None, 128) | 3968 |   
| dense_1 (Dense) | (None, 64) | 8256 |
| dense_2 (Dense) | (None, 32) | 2080 |
| dropout_1 (Dropout) | (None, 32) | 0 |
| dense_3 (Dense) | (None, 1) | 33 |
 
#### Total params: 14,337
#### Trainable params: 14,337
#### Non-trainable params: 0

Pipeline for df_sewing:
| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| input_1 (InputLayer) | [(None, 30)] | 0 |                                                                
| dense (Dense) | (None, 128) | 3968 |   
| dense_1 (Dense) | (None, 64) | 8256 |
| dense_2 (Dense) | (None, 32) | 2080 |
| dropout_1 (Dropout) | (None, 32) | 0 |
| dense_3 (Dense) | (None, 1) | 33 |

#### Total params: 14,337
#### Trainable params: 14,337
#### Non-trainable params: 0

Pipeline for df_fin:
| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| input_1 (InputLayer) | [(None, 30)] | 0 |                                                                
| dense (Dense) | (None, 128) | 3968 |   
| dense_1 (Dense) | (None, 64) | 8256 |
| dense_2 (Dense) | (None, 32) | 2080 |
| dropout_1 (Dropout) | (None, 32) | 0 |
| dense_3 (Dense) | (None, 1) | 33 |

#### Total params: 14,337
#### Trainable params: 14,337
#### Non-trainable params: 0

d. All dataset are trained on a separate instance with validation split of 25%, batch size of 64, epochs of 100 with early stopping patience of 15.

## 3. Evaluation
### 3.1 Whole Dataset
a. Training is completed with 74 epochs with EarlyStopping with best weight on epoch 59

