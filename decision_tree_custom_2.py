import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
    if len(y_true) == 0 or np.all(y_true == 0):
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100  # Avoid division by zero

class DecisionTreeRegressorCustom:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_split:
            return None, None

        best_feature, best_threshold, best_mse = None, None, float('inf')

        for feature_index in range(n_features):
            sorted_indices = np.argsort(X[:, feature_index])
            X_sorted, y_sorted = X[sorted_indices], y[sorted_indices]
            cumulative_sum = np.cumsum(y_sorted)
            cumulative_sq_sum = np.cumsum(y_sorted ** 2)

            for i in range(1, n_samples):
                if X_sorted[i, feature_index] == X_sorted[i - 1, feature_index]:
                    continue

                n_left = i
                n_right = n_samples - i
                if n_left < self.min_samples_split or n_right < self.min_samples_split:
                    continue

                sum_left = cumulative_sum[i - 1]
                sum_right = cumulative_sum[-1] - sum_left
                sq_sum_left = cumulative_sq_sum[i - 1]
                sq_sum_right = cumulative_sq_sum[-1] - sq_sum_left

                mse_left = (sq_sum_left - (sum_left ** 2) / n_left) / n_left
                mse_right = (sq_sum_right - (sum_right ** 2) / n_right) / n_right
                mse_split = (n_left * mse_left + n_right * mse_right) / n_samples

                if mse_split < best_mse:
                    threshold = (X_sorted[i, feature_index] + X_sorted[i - 1, feature_index]) / 2
                    best_feature, best_threshold, best_mse = feature_index, threshold, mse_split

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if len(y) == 0:
            return np.nan  # Handle empty datasets gracefully

        if self.max_depth is not None and depth >= self.max_depth:
            return np.mean(y)
        if len(y) < self.min_samples_split or self._mse(y) == 0:
            return np.mean(y)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, sample, tree):
        while isinstance(tree, dict):
            if sample[tree["feature_index"]] < tree["threshold"]:
                tree = tree["left"]
            else:
                tree = tree["right"]
        return np.nan if tree is None else tree

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

def print_tree(tree, depth=0, max_depth=3):
    if depth > max_depth:
        return
    if isinstance(tree, dict):
        print("  " * depth + f"[Depth {depth}] Feature_{tree['feature_index']} < {tree['threshold']:.2f}")
        print_tree(tree['left'], depth + 1, max_depth)
        print_tree(tree['right'], depth + 1, max_depth)
    else:
        print("  " * depth + f"Prediction: {tree:.2f}")

# Testing the refactored tree with custom metrics
if __name__ == "__main__":
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
        tree = DecisionTreeRegressorCustom(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_results = []
        mae_results = []
        mape_results = []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_val)

            mse = calculate_mse(y_val, y_pred)
            mae = calculate_mae(y_val, y_pred)
            mape = calculate_mape(y_val, y_pred)

            mse_results.append(mse)
            mae_results.append(mae)
            mape_results.append(mape)

        avg_mse = np.mean(mse_results)
        avg_mae = np.mean(mae_results)
        avg_mape = np.mean(mape_results)

        print(f"Params: {params} | Avg MSE: {avg_mse:.2f} | Avg MAE: {avg_mae:.2f} | Avg MAPE: {avg_mape:.2f}%")

        tree.fit(X, y)  # Train the tree
        print("Custom Decision Tree Visualization (First 3 Layers):")
        print_tree(tree.tree, max_depth=3)  # Visualize up to 3 layers
