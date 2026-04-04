import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
import kagglehub
from genericpath import isfile
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error,get_scorer_names
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

warnings.filterwarnings('ignore')

df3 = pd.read_csv('Dataset3.csv', engine='python')
print(df3.shape)

columns_to_drop = ['id', 'date', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']
df3 = df3.drop(columns=[col for col in columns_to_drop if col in df3.columns])
print(df3.columns.tolist())
print(df3.shape)

df3.dropna(inplace=True)
target_column = 'price'
numerical_features = [col for col in df3.columns if col != target_column]
scaler = StandardScaler()
df3[numerical_features] = scaler.fit_transform(df3[numerical_features])
categorical_cols_to_encode = df3.select_dtypes(include='object').columns
df3 = pd.get_dummies(df3, columns=categorical_cols_to_encode, drop_first=True)
features = [col for col in df3.columns if col != target_column]
X = df3[features].copy()
y = df3[target_column].copy()
print("=" * 60)
print("DATASET 3")
print("=" * 60)

print("\nFirst 10 rows:")
display(df3.head(10))

print("\nData types:")
print(df3.dtypes)

print("\nMissing values:")
print(df3.isnull().sum())

print("\nTarget variable (price) statistics:")
print(df3['price'].describe())

nrow,ncol = X.shape
print("the dateset contains ",nrow," records with ",ncol,"features")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

model = LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("Before cross validation")
print(" MSE", metrics.mean_squared_error(y_test, predictions))
print(" RMAE", metrics.root_mean_squared_error(y_test, predictions))
print(" r2", metrics.r2_score(y_test, predictions))

cv_results = cross_validate(model,X_train,y_train,cv=5,scoring=('neg_mean_squared_error','neg_root_mean_squared_error','r2'),return_train_score=True)
print("After cross validation")
print(" MSE:", -cv_results['test_neg_mean_squared_error'].mean())
print(" RMAE",-cv_results['test_neg_root_mean_squared_error'].mean())
print(" r2",cv_results['test_r2'].mean())


plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.4, s=15)
max_val = max(y_test.max(), predictions.max())
plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')

plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Baseline Linear Regression: Predicted vs Actual')
plt.legend()
plt.tight_layout()
plt.show()
