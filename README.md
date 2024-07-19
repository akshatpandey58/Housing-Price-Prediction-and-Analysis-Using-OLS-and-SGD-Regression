# README for Housing Price Prediction Program

## Overview

This program predicts housing prices using two different regression models: Ordinary Least Squares (OLS) and Stochastic Gradient Descent (SGD). The dataset includes various features about houses, such as area, number of bedrooms, bathrooms, and other amenities. The program performs the following steps:

1. Loads and preprocesses the dataset.
2. Splits the data into training and testing sets.
3. Trains and evaluates an OLS regression model.
4. Trains and evaluates an SGD regression model.
5. Makes predictions for a new unseen record.

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Instructions

### 1. Download and Load the Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df_house = pd.read_csv('path_to_downloaded_file/housing.csv')
```

### 2. Preprocess the Dataset

Identify and map binary features:

```python
# Map binary categorical features
binary_map = {'yes': 1, 'no': 0}
binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

for feature in binary_features:
    df_house[feature] = df_house[feature].map(binary_map)
```

Create dummy variables for non-binary categorical features:

```python
# Create dummy variables for 'furnishingstatus'
df_house = pd.get_dummies(df_house, columns=['furnishingstatus'], drop_first=True)
```

### 3. Split the Data into Training and Testing Sets

```python
# Split the data
X = df_house.drop('price', axis=1)
y = df_house['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### 4. Train and Evaluate the OLS Regression Model

Add a constant to input features:

```python
# Add constant to input features
X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)
```

Fit the OLS model:

```python
# Declare the OLS model
model_ols = sm.OLS(y_train, X_train_ols).fit()

# Predict the Test set
y_pred_ols = model_ols.predict(X_test_ols)

# Evaluate the model
mse_ols = mean_squared_error(y_test, y_pred_ols)
print("Mean Square Error (OLS):", mse_ols)
```

Visualize the error:

```python
# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ols, alpha=0.7, color='b')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (OLS)')
plt.show()
```

### 5. Train and Evaluate the SGD Regressor Model

Fit the SGD model:

```python
# Declare the SGDRegressor model
model_sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

# Fit the model
model_sgd.fit(X_train, y_train)

# Predict the Test set
y_pred_sgd = model_sgd.predict(X_test)

# Evaluate the model
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
print("Mean Square Error (SGD):", mse_sgd)
```

Visualize the error:

```python
# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_sgd, alpha=0.7, color='r')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (SGD)')
plt.show()
```

### 6. Prediction for a New Unseen Record

```python
# Generate a new record with input features
new_record = pd.DataFrame({
    'const': [1.0],
    'area': [3000],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad': [1],
    'guestroom': [0],
    'basement': [0],
    'hotwaterheating': [0],
    'airconditioning': [1],
    'parking': [1],
    'prefarea': [1],
    'furnishingstatus_semi-furnished': [0],
    'furnishingstatus_unfurnished': [1]
})

# Predict the sales output using trained OLS model
new_prediction = model_ols.predict(new_record)
print("Predicted Price for the new record (OLS):", new_prediction[0])

# Predict the sales output using trained SGD model
new_prediction_sgd = model_sgd.predict(new_record.drop('const', axis=1))
print("Predicted Price for the new record (SGD):", new_prediction_sgd[0])
```

## Notes

- Ensure that the dataset is correctly loaded from the specified path.
- Modify the paths and parameters as needed.
- The script handles mapping of binary features and creation of dummy variables for categorical features.
- Evaluation metrics and visualizations help in assessing model performance.

By following these steps, you can successfully predict housing prices using both OLS and SGD regression models.
