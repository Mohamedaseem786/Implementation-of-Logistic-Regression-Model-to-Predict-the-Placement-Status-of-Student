# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries such as pandas module to read the corresponding csv file.
2. Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the corresponding dataset values.
4. Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y and Predict the values of array using the variable y_pred.
5. Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
6. Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Karthikeyan.K
RegisterNumber: 212221230046
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![op1](https://user-images.githubusercontent.com/93427303/196496169-0062a22b-10d4-4a21-a0d1-8421f05b05e3.png)

![op2](https://user-images.githubusercontent.com/93427303/196496187-98b4c622-ab67-47f9-86cd-1f38c3882ff2.png)

![op3](https://user-images.githubusercontent.com/93427303/196496206-8c8c1748-2ce8-4519-a407-c7415c51155a.png)

![op4](https://user-images.githubusercontent.com/93427303/196496219-6ef1dceb-de0b-43ff-9380-2edea07b03e7.png)

![op5](https://user-images.githubusercontent.com/93427303/196496253-4de01319-dcbf-4586-86f8-bea5fda1ad13.png)

![op6](https://user-images.githubusercontent.com/93427303/196496270-0110adcf-b7cc-4405-be1b-1f0b08c26b0c.png)

![op7](https://user-images.githubusercontent.com/93427303/196496291-b85cd873-a4c2-4522-bcff-15e0b187adcd.png)

![op8](https://user-images.githubusercontent.com/93427303/196496437-2d4aa158-e3e9-4bc9-a806-062d6f17dbfe.png)

![op9](https://user-images.githubusercontent.com/93427303/196496559-deff4ac0-2f49-42a2-8c65-248f3c0bacf4.png)

![op10](https://user-images.githubusercontent.com/93427303/196496574-665950b8-1c42-44d5-bb45-0150222fcf07.png)

![op11](https://user-images.githubusercontent.com/93427303/196496591-28c92d04-a216-45bf-be5c-99aa07b0fb48.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
