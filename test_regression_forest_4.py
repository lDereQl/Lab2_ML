import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from random_forest_regression_custom_4 import RandomForestRegressorCustom
from sklearn.ensemble import RandomForestRegressor

# Load and clean data
data = pd.read_csv("train_data_final.csv")
data_test = pd.read_csv("test_data_final.csv")

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(data.drop(columns=["cnt"]))
y_train = data["cnt"].values
X_test = imputer.transform(data_test.drop(columns=["cnt"]))
y_test = data_test["cnt"].values

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

# Check for NaN in predictions
if np.any(np.isnan(y_pred_custom)):
    print("Warning: NaN values in custom predictions. Replacing with 0.")
    y_pred_custom = np.nan_to_num(y_pred_custom)

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
    """Interactive input functionality for user predictions."""
    if not hasattr(model_custom, 'n_features') or model_custom.n_features is None:
        print("Error: Custom model is not trained. Train the model before making predictions.")
        return

    expected_features = model_custom.n_features
    print(f"\nEnter {expected_features} feature values for prediction (comma-separated):")
    try:
        # Take user input and convert to numpy array
        user_input = input("Enter values: ")
        features = np.array([float(x) for x in user_input.split(",")]).reshape(1, -1)

        # Check if input matches the expected number of features
        if features.shape[1] != expected_features:
            print(f"Error: Expected {expected_features} features but received {features.shape[1]}.")
            return

        # Predictions
        pred_custom = model_custom.predict(features)[0]
        pred_sklearn = model_sklearn.predict(features)[0]

        # Display predictions
        print(f"Custom Model Prediction: {pred_custom:.2f}")
        print(f"Scikit-learn Model Prediction: {pred_sklearn:.2f}")
    except ValueError:
        print("Invalid input! Ensure all values are numeric.")


# Interactive Input
predict_input(custom_rf, sklearn_rf)
