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

c. The pipeline is the same for all dataset. Each dense layer uses the 'elu' activation function.

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
| input_1 (InputLayer) | [(None, 29)] | 0 |                                                                
| dense (Dense) | (None, 128) | 3840 |   
| dense_1 (Dense) | (None, 64) | 8256 |
| dense_2 (Dense) | (None, 32) | 2080 |
| dropout_1 (Dropout) | (None, 32) | 0 |
| dense_3 (Dense) | (None, 1) | 33 |

#### Total params: 14,209
#### Trainable params: 14,209
#### Non-trainable params: 0

Pipeline for df_fin:
| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| input_1 (InputLayer) | [(None, 25)] | 0 |                                                                
| dense (Dense) | (None, 128) | 3328 |   
| dense_1 (Dense) | (None, 64) | 8256 |
| dense_2 (Dense) | (None, 32) | 2080 |
| dropout_1 (Dropout) | (None, 32) | 0 |
| dense_3 (Dense) | (None, 1) | 33 |

#### Total params: 13,697
#### Trainable params: 13,697
#### Non-trainable params: 0

d. All dataset are trained on a separate instance with validation split of 25%, batch size of 64, epochs of 100 with early stopping patience of 15. The loss is optimized using 'adam'.

## 3. Evaluation
### 3.1 Complete Dataset
a. Training is completed with 74 epochs with EarlyStopping with best weight on epoch 59

![image](https://user-images.githubusercontent.com/82880708/181160261-0edd9436-9ae7-4310-8a6a-0291fc679669.png)
![image](https://user-images.githubusercontent.com/82880708/181160276-c988514a-325a-44b0-b2b3-235bd3f152d0.png)
![image](https://user-images.githubusercontent.com/82880708/181160376-c86d3d55-344d-4846-a351-15f9c3619d6e.png)

b. Mean Squared Error: 0.022068, Mean Absolute error: 0.107073

c. Correlation Coefficient of predicted productivity vs actual productivity is 0.53. This indicates a moderate strength of correlation. A visible trendline of y = x can be seen from the graph above.

### 3.2 'sewing' Department Dataset
a. Training is completed with 70 epochs with EarlyStopping with best weight on epoch 55

![image](https://user-images.githubusercontent.com/82880708/181161218-6333a21c-03a2-4a20-bb86-d23d05f7b53a.png)
![image](https://user-images.githubusercontent.com/82880708/181161239-205663a5-70a6-497a-9b50-cafb7d703d3a.png)
![image](https://user-images.githubusercontent.com/82880708/181161247-9d59d601-0b8c-437c-a911-87632c417341.png)

b. Mean Squared Error: 0.008375, Mean Absolute error: 0.063756

c. Correlation Coefficient of predicted productivity vs actual productivity is 0.81. This indicates a very strong strength of correlation. A very clear trendline of y = x can be seen from the graph above

### 3.3 'finishing' Department Dataset
a. Training is completed with 29 epochs with EarlyStopping with best weight on epoch 14

![image](https://user-images.githubusercontent.com/82880708/181161962-47833126-0770-49ac-8d12-9fda5c50c85d.png)
![image](https://user-images.githubusercontent.com/82880708/181161971-f1d1e0ec-88a5-4275-8e2f-c665453c2791.png)
![image](https://user-images.githubusercontent.com/82880708/181161976-f328816e-4468-4862-a7a9-7ecffd08a5ea.png)

b. Mean Squared Error: 0.044979, Mean Absolute error: 0.164808

c. Correlation Coefficient of predicted productivity vs actual productivity is 0.34. This indicates a weak strength of correlation. A somewhat visible trendline of y = x can be seen from the graph above.

## 4. Conclusion
|Datasets|Mean Squared Error|Mean Absolute error|Correlation Coefficient of Prediction vs Actual|Relationship|
|---|---|---|---|---|
|Complete|0.022068|0.063756|0.53|Moderate|
|Sewing Department|0.008375|0.063756|0.81|Strong|
|Finishing Department|0.044979|0.164808|0.34|Weak|

#### We should choose different model depending on the department. As we can see from above, using the model generated from 'sewing' department gives a more accurate prediction. However, the 'finishing' department model requires additional training.
