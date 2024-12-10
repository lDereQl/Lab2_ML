from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


# Custom metric implementations
def calculate_mse(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    return np.mean((y_true - y_pred) ** 2)


def calculate_mae(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true, y_pred):
    if len(y_true) == 0 or np.any(y_true == 0):
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Perform cross-validation for both implementations
def cross_validate_model(model, X, y, param_combinations, model_name="Custom"):
    results = []

    for params in param_combinations:
        # Apply parameters to the model
        if model_name == "Custom":
            model_instance = model(**params)
        else:
            model_instance = model(**params, random_state=42)

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_results, mae_results, mape_results = [], [], []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_val)

            mse = calculate_mse(y_val, y_pred)
            mae = calculate_mae(y_val, y_pred)
            mape = calculate_mape(y_val, y_pred)

            mse_results.append(mse)
            mae_results.append(mae)
            mape_results.append(mape)

        avg_mse = np.mean(mse_results)
        avg_mae = np.mean(mae_results)
        avg_mape = np.mean(mape_results)

        results.append({
            "Model": model_name,
            "Params": params,
            "Avg MSE": avg_mse,
            "Avg MAE": avg_mae,
            "Avg MAPE": avg_mape
        })

    return pd.DataFrame(results)


# Load data
data = pd.read_csv("train_data_final.csv")
X = data.drop(columns=["cnt"]).values
y = data["cnt"].values

# Parameter combinations
param_combinations = [
    {"max_depth": 5, "min_samples_split": 10},
    {"max_depth": 10, "min_samples_split": 20},
    {"max_depth": 15, "min_samples_split": 10},
    {"max_depth": 10, "min_samples_split": 50},
    {"max_depth": 13, "min_samples_split": 10},
]

# Import Custom Decision Tree
from decision_tree_custom_2 import DecisionTreeRegressorCustom

# Cross-validate custom implementation
custom_results = cross_validate_model(DecisionTreeRegressorCustom, X, y, param_combinations, model_name="Custom")

# Cross-validate Scikit-learn implementation
sklearn_results = cross_validate_model(DecisionTreeRegressor, X, y, param_combinations, model_name="Scikit-learn")

# Combine and save results
all_results = pd.concat([custom_results, sklearn_results], ignore_index=True)
all_results.to_csv("cross_validation_results.csv", index=False)

print("Cross-validation results saved to 'cross_validation_results.csv'.")
