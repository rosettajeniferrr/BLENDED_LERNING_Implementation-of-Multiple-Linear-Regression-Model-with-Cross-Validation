# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import required libraries and load the dataset.
2.  Remove unnecessary columns and convert categorical data using dummy variables.
3.  Separate the dataset into features (X) and target variable (Y).
4.  Split the data into training and testing sets.
5.  Create and train the Linear Regression model using training data.
6.  Evaluate the model using 5-fold cross-validation and calculate average R² score.
7.  Predict test data values and compute MSE, MAE, and R²; plot actual vs predicted prices.

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Rosetta Jenifer C
RegisterNumber:  212225230230
*/
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('CarPrice_Assignment.csv')

data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)

#split data
X=data.drop('price',axis=1)
Y=data['price']
X_train,X_test,Y_train,Y_test=train_test_split = train_test_split(X,Y,test_size=0.2,random_state=42)

#create & train model
model = LinearRegression()
model.fit(X_train,Y_train)

#evaluate with cross validation 
print('Name:Rosetta Jenifer C')
print('Reg No: 212225230230')
print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(model,X,Y,cv=5)
print("Fold R2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")

Y_pred = model.predict(X_test)
print("\n=== Test Set Performance ===")
print('MSE:', mean_squared_error(Y_test, Y_pred))
print('MAE:', mean_absolute_error(Y_test, Y_pred))
print('R2 Score =', r2_score(Y_test, Y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()],[Y_test.min(), Y_test.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()

```

## Output:
<img width="796" height="171" alt="image" src="https://github.com/user-attachments/assets/9e4d74f5-64b7-4487-ac8a-ee868fc70b53" />
<img width="558" height="153" alt="image" src="https://github.com/user-attachments/assets/9bf226e4-fa71-4667-a83e-d7c9144b06e1" />
<img width="1352" height="771" alt="image" src="https://github.com/user-attachments/assets/d6039ec5-fcba-4634-980d-09f41514eb96" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
