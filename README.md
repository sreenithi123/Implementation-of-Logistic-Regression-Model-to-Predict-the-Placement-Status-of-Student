
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
``
1. 1 Import the required packages and print the present data.
2  Print the placement data and salary data.
3  Find the null and duplicate values.
4  Using logistic regression find the predicted values of accuracy , confusion matrices.
5  Display the results
```

   

## Program:
```
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sreenithi.E
RegisterNumber:  212223220109
*/
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
````
``



 
*/

## Output:
PLACEMENT DATA

![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/d1bdd9f6-f3ee-472b-ab30-185af470aaa1)

SALARY DATA

![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/9acc51bd-bb31-4d53-8c61-03022a550d73)

![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/70975a43-6cde-4f5a-8c47-6c7a92e38a4b)

CHECKING THE NULL() FUNTION

![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/1eae1d9f-d6f9-4cd7-954d-91e58618ea57)



DATA DUPLICATE

![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/65dc166c-bc47-4733-b3cf-72ffcb5dbb20)
PRINT DATA

![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/96494ead-c8d6-42b5-8ad7-6c533de411e9)
DATA-STATUES


![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/289e06db-ef49-437f-babb-7f071baa6df7)

Y_prediction array:


![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/c7beeda6-1d0c-4544-90e0-e83b622dff6f)


Accuracy value:


![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/e9485eef-3966-4a26-bf8d-0e8c3c7a7db6)


Confusion array:

![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/6a10e3fe-0542-4ee7-b3f0-d0be44e8e097)


Classification Report


![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/6f748139-f92d-4e84-ab6d-824549842b70)

Prediction of LR: 

![image](https://github.com/sreenithi123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743046/1fa4dae3-65be-4ee3-ac33-5b01ad271402)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
