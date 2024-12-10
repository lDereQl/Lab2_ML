from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

# Custom MAPE implementation
def calculate_mape(y_true, y_pred):
    if len(y_true) == 0 or np.any(y_true == 0):
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load data
data = pd.read_csv("train_data_final.csv")
X = data.drop(columns=["cnt"]).values
y = data["cnt"].values

# Parameter combinations for testing
param_combinations = [
    {"max_depth": 5, "min_samples_split": 10},
    {"max_depth": 10, "min_samples_split": 50},
    {"max_depth": 10, "min_samples_split": 20},
    {"max_depth": 13, "min_samples_split": 10},
    {"max_depth": 15, "min_samples_split": 10},
]

# Evaluate using cross-validation
for params in param_combinations:
    tree = DecisionTreeRegressor(**params, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_results = []
    mae_results = []
    mape_results = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mape = calculate_mape(y_val, y_pred)

        mse_results.append(mse)
        mae_results.append(mae)
        mape_results.append(mape)

    avg_mse = np.mean(mse_results)
    avg_mae = np.mean(mae_results)
    avg_mape = np.mean(mape_results)

    print(f"Params: {params} | Avg MSE: {avg_mse:.2f} | Avg MAE: {avg_mae:.2f} | Avg MAPE: {avg_mape:.2f}%")

# Visualize the best-performing tree (example visualization for max_depth=10, min_samples_split=20)
best_tree = DecisionTreeRegressor(max_depth=10, min_samples_split=20, random_state=42)
best_tree.fit(X, y)

plt.figure(figsize=(20, 10))
plot_tree(best_tree, feature_names=data.drop(columns=["cnt"]).columns, max_depth=3, filled=True)
plt.title("Scikit-learn Decision Tree (First 3 Layers)")
plt.show()
