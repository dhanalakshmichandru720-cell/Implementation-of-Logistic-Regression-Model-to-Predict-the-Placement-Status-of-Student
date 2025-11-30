# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
/*
# Program to implement the Logistic Regression Model to predict the Placement Status of Students
# Developed by: Dhanalakshmi.C
# RegisterNumber: 25A018616

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Suppose we have: CGPA, Aptitude Score, Communication Skill → Placement (1=Placed, 0=Not Placed)

data = {
    'CGPA': [8.5, 6.8, 7.9, 5.4, 9.1, 8.0, 7.5, 6.0, 9.3, 5.8],
    'Aptitude_Score': [82, 55, 75, 48, 92, 77, 73, 50, 95, 45],
    'Communication_Skill': [8, 6, 7, 5, 9, 8, 7, 6, 9, 5],
    'Placed': [1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")

X = df[['CGPA', 'Aptitude_Score', 'Communication_Skill']]
y = df['Placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

new_student = np.array([[8.2, 78, 8]])   # CGPA, Aptitude, Communication
prediction = model.predict(new_student)

print("\nNew Student Prediction:")
if prediction[0] == 1:
    print("The student is likely to be PLACED.")
else:
    print("The student is NOT likely to be placed.")



##output 
dataset:
   CGPA  Aptitude_Score  Communication_Skill  Placed
0   8.5              82                    8       1
1   6.8              55                    6       0
2   7.9              75                    7       1
3   5.4              48                    5       0
4   9.1              92                    9       1
5   8.0              77                    8       1
6   7.5              73                    7       1
7   6.0              50                    6       1
8   9.3              95                    9       1
9   5.8              45                    5       0


Confusion Matrix:
[[2 0]
 [1 0]]

Classification Report:
              precision    recall  f1-score   support

           0       0.67      1.00      0.80         2
           1       0.00      0.00      0.00         1

    accuracy                           0.67         3
   macro avg       0.33      0.50      0.40         3
weighted avg       0.44      0.67      0.53         3

Accuracy: 0.6666666666666666

New Student Prediction:
The student is likely to be PLACED.

""
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
