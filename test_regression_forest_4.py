import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from random_forest_custom_4 import RandomForestRegressorCustom

from decision_tree_custom_2 import DecisionTreeRegressorCustom

# Load California Housing dataset
# Data placeholders (replace with actual dataset)
data = pd.read_csv("train_data_final.csv")
if data.isnull().sum().any():
    print("Dataset contains NaN values. Handling missing data.")
    data = data.dropna()  # Or handle missing data with an imputation strategy
data_test= pd.read_csv("test_data_final.csv")
if data_test.isnull().sum().any():
    print("Dataset contains NaN values. Handling missing data.")
    data = data.dropna()  # Or handle missing data with an imputation strategy
X_train = data.drop(columns=["cnt"]).values
y_train = data["cnt"].values
X_test = data_test.drop(columns=["cnt"]).values
y_test = data_test["cnt"].values

# Initialize Custom Random Forest
model = RandomForestRegressorCustom(
    n_estimators=10,  # Number of trees
    max_depth=5,  # Maximum depth of each tree
    min_samples_split=2,  # Minimum samples to split
    max_features="sqrt",  # Feature selection strategy
    random_state=42  # For reproducibility
)


# Progress Bar Enhancement
def fit_with_progress(model, X, y):
    """Fits the Random Forest with a progress bar."""
    np.random.seed(model.random_state)
    model.trees = []

    for _ in tqdm(range(model.n_estimators), desc="Training Random Forest"):
        X_sample, y_sample = model._bootstrap_sample(X, y)
        feature_indices = model._feature_subset(X_sample)

        tree = DecisionTreeRegressorCustom(
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            random_state=model.random_state
        )
        tree.fit(X_sample[:, feature_indices], y_sample)
        model.trees.append((tree, feature_indices))


# Use the updated fit method
model.fit = lambda X, y: fit_with_progress(model, X, y)

# Train the model
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
