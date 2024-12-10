from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

# Load the dataset
train_data = pd.read_csv("train_data_final.csv")
val_data = pd.read_csv("val_data_final.csv")

X_train = train_data.drop(columns=["cnt"]).values
y_train = train_data["cnt"].values

X_val = val_data.drop(columns=["cnt"]).values
y_val = val_data["cnt"].values

# Train the Scikit-learn Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=20,  # Number of trees
    max_depth=15,  # Maximum depth of each tree
    random_state=42  # Seed for reproducibility
)
rf.fit(X_train, y_train)

# Predict on validation data
y_pred = rf.predict(X_val)

# Evaluate metrics
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-5))) * 100  # Avoid division by zero

# Print metrics
print("Scikit-learn Random Forest Regression Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
