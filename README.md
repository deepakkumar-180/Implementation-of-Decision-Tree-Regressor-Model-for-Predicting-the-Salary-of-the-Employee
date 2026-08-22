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
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv("employee_salary.csv")

# Independent variable
X = data[["Level"]]

# Dependent variable
y = data["Salary"]

# Create Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X, y)

# Get level from user
level = float(input("Enter employee level: "))

# Predict salary
prediction = model.predict([[level]])

print("Predicted Salary:", prediction[0])

# Plot the Decision Tree Regression
plt.scatter(X, y, color="red")
plt.plot(X, model.predict(X), color="blue")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Decision Tree Regression - Employee Salary")
plt.show()

```

## Output:

<img width="1477" height="797" alt="image" src="https://github.com/user-attachments/assets/28aa36d9-9718-4035-9d4d-8803291af926" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
