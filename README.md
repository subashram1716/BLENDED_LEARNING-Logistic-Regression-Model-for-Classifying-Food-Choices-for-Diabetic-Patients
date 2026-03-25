# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data Import and prepare the dataset to initiate the analysis workflow.

2.Explore Data Examine the data to understand key patterns, distributions, and feature relationships.

3.Select Features Choose the most impactful features to improve model accuracy and reduce complexity.

4.Split Data Partition the dataset into training and testing sets for validation purposes.

5.Scale Features Normalize feature values to maintain consistent scales, ensuring stability during training.

6.Train Model with Hyperparameter Tuning Fit the model to the training data while adjusting hyperparameters to enhance performance.

7.Evaluate Model Assess the model’s accuracy and effectiveness on the testing set using performance metrics.

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: SUBASHRAM T
RegisterNumber: 212225040430
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("food_items (1) (1).csv")
print("Dataset Overview")
print(df.head())
print("\ndatset Info")
print(df.info())

X_raw=df.iloc[:, :-1]
y_raw=df.iloc[:, -1:]
X_raw

scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

penalty='l2'
multi_class='multnomial'
solver='lbfgs'
max_iter=1000

model = LogisticRegression(max_iter=2000) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
*/
```

## Output:
<img width="1241" height="619" alt="Screenshot 2026-03-25 132026" src="https://github.com/user-attachments/assets/47fc1a67-af53-46d4-847f-74967775653d" />
<img width="1228" height="585" alt="Screenshot 2026-03-25 132104" src="https://github.com/user-attachments/assets/05652a66-98a1-4195-9204-ae598e76706d" />
<img width="1246" height="348" alt="Screenshot 2026-03-25 132142" src="https://github.com/user-attachments/assets/3daf9f4e-771b-4e4f-ba38-850d6b05a4de" />
<img width="1232" height="498" alt="Screenshot 2026-03-25 132201" src="https://github.com/user-attachments/assets/4a9f04e0-2585-436c-88ae-d73c7f3eaea6" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
