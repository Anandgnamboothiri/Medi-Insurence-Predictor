import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#LOADING DATA(THE INSURANCE DATA)
data = pd.read_csv('medical insurance charge pridiction/archive/insurance_new.csv')
data_df = pd.DataFrame(data)

#ANALYSING AND VISUALIZING USING PANDAS AND MATPLOTLIB

data_df.info()
data_df.describe()

plt.figure(figsize=(10,6))
plt.hist(data_df['age'],bins=40,color='blue',edgecolor='black')
plt.xlabel('age')
plt.ylabel('count')
plt.show()

X = data_df.drop('charges', axis=1)
y = data_df['charges']

# 1. Separate categorical and numerical columns
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']

# 2. One-hot encode categorical
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cats = encoder.fit_transform(data_df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# 3. Combine numerical and encoded categorical
full_features = pd.concat([data_df[numerical_cols], encoded_df], axis=1)

# 4. Scale the final features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(full_features)


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,train_size=0.8,random_state=100)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# 🧠 Step 2: Fit to training data
model.fit(x_train, y_train)

# 🧠 Step 3: Predict on test data
y_pred = model.predict(x_test)





# 🧠 Step 4: Visualize Actual vs Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.legend()
plt.show()

# 🧠 Step 5: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10,6))
plt.scatter(y_pred, residuals, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

# 🧠 Step 6: Evaluate Mode
#l Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")


