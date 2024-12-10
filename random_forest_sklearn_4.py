import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data = pd.read_csv("train_data_final.csv")
val_data = pd.read_csv("val_data_final.csv")

X_train = data.drop(columns=["cnt"]).values
y_train = data["cnt"].values

X_val = val_data.drop(columns=["cnt"]).values
y_val = val_data["cnt"].values

# Hyperparameter tuning
n_estimators_list = [5, 10, 15]
max_depth_list = [5, 10, 15]
min_samples_split_list = [10, 20, 50]

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            fold_mse = []
            for train_index, val_index in kf.split(X_train):
                X_tr, X_val = X_train[train_index], X_train[val_index]
                y_tr, y_val = y_train[train_index], y_train[val_index]

                # Train Scikit-learn Random Forest
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                rf.fit(X_tr, y_tr)

                # Predict and calculate MSE
                y_pred = rf.predict(X_val)
                mse = np.mean((y_val - y_pred) ** 2)
                fold_mse.append(mse)

            # Store average MSE for the parameter combination
            avg_mse = np.mean(fold_mse)
            results.append((n_estimators, max_depth, min_samples_split, avg_mse))

# Sort results by MSE
results.sort(key=lambda x: x[3])

# Print best parameters
best_params = results[0]
print(f"Best Parameters: n_estimators={best_params[0]}, max_depth={best_params[1]}, min_samples_split={best_params[2]} with MSE={best_params[3]:.2f}")

# Visualization: Boxplot for n_estimators
mse_by_estimators = {n: [] for n in n_estimators_list}

for n_estimators, max_depth, min_samples_split, mse in results:
    mse_by_estimators[n_estimators].append(mse)

plt.figure(figsize=(10, 6))
ax = plt.gca()
for n_estimators, mse_values in mse_by_estimators.items():
    ax.boxplot(mse_values, positions=[n_estimators], widths=0.6, tick_labels=[str(n_estimators)])

plt.xlabel("Number of Estimators (n_estimators)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE Distribution Across Different n_estimators")
plt.grid(True)
plt.show()