# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
# Name: DEEPAKKUMAR S
# Reg No: 212225230042
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the employee dataset and separate input features and salary target variable.

2.Split the dataset into training and testing data.

3.Train a Decision Tree Regressor using the training data.

4.Predict the employee salary and calculate the model accuracy.


## Program:

```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Employee dataset
data = {
    'Age': [25, 30, 35, 40, 45, 28, 32, 38, 42, 50],
    'Experience': [1, 3, 5, 8, 12, 2, 4, 7, 10, 15],
    'Education': [12, 14, 16, 16, 18, 14, 15, 16, 18, 18],
    'Salary': [25000, 30000, 40000, 50000, 65000,
               28000, 35000, 45000, 55000, 75000]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Age', 'Experience', 'Education']]
y = df['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Actual Salary   :", y_test.values)
print("Predicted Salary:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict salary for a new employee
new_employee = [[30, 3, 14]]
prediction = model.predict(new_employee)

print("Predicted Salary for New Employee:", prediction[0])

```

## Output:

<img width="935" height="646" alt="image" src="https://github.com/user-attachments/assets/5bc6990b-a664-446f-b1f3-c00e811d1278" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
