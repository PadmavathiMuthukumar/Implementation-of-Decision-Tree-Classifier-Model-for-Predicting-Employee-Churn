# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Prepare your data
2. Define your model
3. Define your cost function
4. Define your learning rate
5. Train your model
6. Evaluate your model
7. Tune hyperparameters
8. Deploy your model
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:  Padmavathi M
RegisterNumber: 212223040141 
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull.sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data[["left"]]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
## Dataset:
![image](https://github.com/user-attachments/assets/c7192a1d-da25-4235-a5e3-b049b241ae51)
## DataInfo:
![image](https://github.com/user-attachments/assets/f3d92ff1-21b8-4536-909c-fbc43fe475c9)
## Labelling
![image](https://github.com/user-attachments/assets/6edbc40f-f861-420d-acd0-35927e160904)
## Assignment of x and y values:
![image](https://github.com/user-attachments/assets/e53e7a29-6204-497d-8b48-e62398a6d789)
![image](https://github.com/user-attachments/assets/38529586-5c25-407c-8d7b-fd3a2207456c)
## Accuracy:
![image](https://github.com/user-attachments/assets/9e01cf1f-49ff-43d5-9d7e-3a8616a2423e)
## Prediction:
![image](https://github.com/user-attachments/assets/b13dd12f-6592-4bf0-a3f9-1c219f5c8a8f)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
