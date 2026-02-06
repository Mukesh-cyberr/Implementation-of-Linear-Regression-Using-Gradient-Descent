# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Initialize Parameters - Set initial values for slope m = 0, y-intercept b = 0, and learning rate α (e.g., 0.01).

Step 2: Load the Dataset - Extract independent variable X (study hours) and dependent variable Y (marks scored).

Step 3: Define the Cost Function - Calculate J(m,b) = (1/2n) Σ(ŷi - yi)² where ŷi = mxi + b.

Step 4: Compute Gradients - Calculate ∂J/∂m = (1/n) Σ(ŷi - yi)xi and ∂J/∂b = (1/n) Σ(ŷi - yi).

Step 5: Update Parameters Using Gradient Descent - Update m = m - α(∂J/∂m) and b = b - α(∂J/∂b), repeat for fixed iterations or until convergence.

Step 6: Form the Linear Equation and Visualize - Obtain final equation Y = mX + b and plot scatter plot with regression line.

## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Mukesh Raj D
RegisterNumber: 212224100038
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:

Data Information

<img width="558" height="222" alt="image" src="https://github.com/user-attachments/assets/e16619ae-fe7f-4005-88eb-43e4b35424a3" />


Value of X

<img width="702" height="706" alt="image" src="https://github.com/user-attachments/assets/350069fa-f5dc-4004-9b7b-363bcd43f031" />


Value of X1_scaled

<img width="589" height="829" alt="image" src="https://github.com/user-attachments/assets/cc9f0265-284f-4176-94e4-a18d26bd7768" />

Predicted Value

<img width="484" height="58" alt="image" src="https://github.com/user-attachments/assets/8daf0e87-c86c-4acd-b9e2-678e9b868958" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
