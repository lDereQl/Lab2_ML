import numpy as np
import pandas as pd
from tqdm import tqdm
from decision_tree_custom_2 import DecisionTreeRegressorCustom


class RandomForestRegressorCustom:
    def __init__(self, n_estimators=50, max_depth=10, min_samples_split=20, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # Number of features to consider for each split
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _get_feature_indices(self, n_features):
        if self.max_features == "sqrt":
            return np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        elif self.max_features == "log2":
            return np.random.choice(n_features, int(np.log2(n_features)), replace=False)
        elif isinstance(self.max_features, int):
            return np.random.choice(n_features, self.max_features, replace=False)
        else:
            return np.arange(n_features)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.trees = []
        self.feature_indices = []

        # Add progress bar
        for _ in tqdm(range(self.n_estimators), desc="Training Random Forest"):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feature_indices = self._get_feature_indices(n_features)
            self.feature_indices.append(feature_indices)

            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeRegressorCustom(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))

        for i, tree in enumerate(self.trees):
            feature_indices = self.feature_indices[i]
            predictions[:, i] = tree.predict(X[:, feature_indices])

        return np.mean(predictions, axis=1)


# Test the Random Forest implementation
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("train_data_final.csv")
    val_data = pd.read_csv("val_data_final.csv")

    X_train = data.drop(columns=["cnt"]).values
    y_train = data["cnt"].values

    X_val = val_data.drop(columns=["cnt"]).values
    y_val = val_data["cnt"].values

    # Train the Random Forest
    rf = RandomForestRegressorCustom(
        n_estimators=50, max_depth=15, min_samples_split=10, max_features="sqrt", random_state=42
    )
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_val)
    mse = np.mean((y_val - y_pred) ** 2)
    mae = np.mean(np.abs(y_val - y_pred))
    mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-5))) * 100  # Adjusted MAPE calculation

    # Print results
    print("Random Forest Regression (Custom) Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
