# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

*/
```

## Output:
## HEAD VALUES
<img width="1090" height="115" alt="319854949-4ddf5d62-c261-42be-8b67-6f6df35f3d36" src="https://github.com/user-attachments/assets/5c16cfef-ae4a-47ea-b41b-1d959ee9aa90" />

## TAIL VALUES
<img width="1090" height="115" alt="319855044-dfa9fd82-e723-4ed4-aaec-ad66ffa348e1" src="https://github.com/user-attachments/assets/0c77947e-dd99-4073-ac79-016bfb309f7f" />

## COMPARE DATASET
<img width="1090" height="461" alt="319855079-9118849a-2323-4b9c-a362-3dc8d060587a" src="https://github.com/user-attachments/assets/e66b07c8-7237-41f6-8a00-44ae68ae6551" />

## PREDICTION VALUE IN X AND Y
<img width="1090" height="55" alt="319855219-5ed7921d-08e1-408e-8383-6899933b01ee" src="https://github.com/user-attachments/assets/f24dc034-6691-43d1-bb6d-62490808b6fc" />

## TRAINING SET
<img width="1090" height="452" alt="319855290-9fae6c7e-e00d-449f-8f16-cbbe94095e76" src="https://github.com/user-attachments/assets/b4d7e836-e0ab-4e46-b257-6edb7b5ee135" />

## TRAINING SET
<img width="1090" height="452" alt="319855335-bea6f3d4-6bf1-4c28-b7ff-167a07ec1d30" src="https://github.com/user-attachments/assets/86b2b80c-2a9b-44f2-bdd0-3fd27de03af1" />

## MSE MAE RMSE
<img width="1090" height="56" alt="319855368-0f1f6538-a2a8-4c7c-838e-8ccbca0c7b6c" src="https://github.com/user-attachments/assets/8f2de20f-e849-4b1b-83a7-0ee20a1473b8" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
