import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from random_forest_custom_4 import RandomForestRegressorCustom
from sklearn.ensemble import RandomForestRegressor

# Load dataset (replace with actual file paths or data loading process)
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Best Parameters
best_params = {
    'n_estimators': 15,
    'max_depth': 15,
    'min_samples_split': 10
}

# Train Custom Random Forest
custom_rf = RandomForestRegressorCustom(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42
)
custom_rf.fit(X_train, y_train)

# Predict using Custom Random Forest
y_pred_custom = custom_rf.predict(X_test)

# Evaluate Custom Random Forest
custom_mse = mean_squared_error(y_test, y_pred_custom)
custom_mae = mean_absolute_error(y_test, y_pred_custom)
custom_r2 = r2_score(y_test, y_pred_custom)

print("Custom Random Forest:")
print(f"MSE: {custom_mse:.2f}")
print(f"MAE: {custom_mae:.2f}")
print(f"R^2: {custom_r2:.2f}")

# Train Scikit-learn Random Forest
sklearn_rf = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42
)
sklearn_rf.fit(X_train, y_train)

# Predict using Scikit-learn Random Forest
y_pred_sklearn = sklearn_rf.predict(X_test)

# Evaluate Scikit-learn Random Forest
sklearn_mse = mean_squared_error(y_test, y_pred_sklearn)
sklearn_mae = mean_absolute_error(y_test, y_pred_sklearn)
sklearn_r2 = r2_score(y_test, y_pred_sklearn)

print("\nScikit-learn Random Forest:")
print(f"MSE: {sklearn_mse:.2f}")
print(f"MAE: {sklearn_mae:.2f}")
print(f"R^2: {sklearn_r2:.2f}")

# Display Predicted vs Actual Values (10 examples)
print("\nPredicted vs Actual Values (Custom vs Sklearn):")
for i in range(10):
    print(f"Actual: {y_test[i]:.2f} | Custom: {y_pred_custom[i]:.2f} | Sklearn: {y_pred_sklearn[i]:.2f}")

# Interactive Input Functionality
def predict_input(model_custom, model_sklearn):
    print("\nEnter feature values for prediction (comma-separated):")
    user_input = input()
    features = np.array([float(x) for x in user_input.split(",")]).reshape(1, -1)

    pred_custom = model_custom.predict(features)[0]
    pred_sklearn = model_sklearn.predict(features)[0]

    print(f"Custom Model Prediction: {pred_custom:.2f}")
    print(f"Scikit-learn Model Prediction: {pred_sklearn:.2f}")


