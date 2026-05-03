# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset.
2. Split and normalize data.
3. Train SGD Classifier.
4. Predict and check accuracy.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Placement_Data.csv")
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
m = len(y)
X = np.c_[np.ones(m), X]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-10, 1 - 1e-10)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []
for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    cost = cost_function(X, y, theta)
    cost_history.append(cost)
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100
print("Accuracy:", accuracy)
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Time")
plt.show()
Developed by: Ritika S
RegisterNumber: 212225220086
*/
```
## Output:
![prediction of iris species using SGD Classifier](sam.png)
<img width="604" height="811" alt="Screenshot 2026-05-03 213300" src="https://github.com/user-attachments/assets/9457d118-b5d2-4bd7-b007-3d7452ee46be" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
