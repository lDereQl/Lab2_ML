import numpy as np
import pandas as pd
from classification_tree_custom_3 import ClassificationTreeCustom
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class RandomForestClassifierCustom:
    def __init__(self, n_estimators=50, max_depth=15, min_samples_split=10, max_features="sqrt", random_state=None):
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

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feature_indices = self._get_feature_indices(n_features)
            self.feature_indices.append(feature_indices)

            # Train a decision tree on the bootstrap sample
            tree = ClassificationTreeCustom(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        all_predictions = np.zeros((X.shape[0], self.n_estimators), dtype=int)

        for i, tree in enumerate(self.trees):
            feature_indices = self.feature_indices[i]
            all_predictions[:, i] = tree.predict(X[:, feature_indices])

        # Aggregate predictions using majority voting
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=all_predictions
        )
        return final_predictions


# Test the Random Forest implementation
if __name__ == "__main__":
    # Load dataset
    train_data = pd.read_csv("train_data_with_classes.csv")
    val_data = pd.read_csv("val_data_with_classes.csv")

    X_train = train_data.drop(columns=["cnt", "cnt_class"]).values
    y_train = train_data["cnt_class"].values.astype(int)

    X_val = val_data.drop(columns=["cnt", "cnt_class"]).values
    y_val = val_data["cnt_class"].values.astype(int)

    # Train the Random Forest
    rf = RandomForestClassifierCustom(
        n_estimators=50, max_depth=10, min_samples_split=20, max_features="sqrt", random_state=42
    )
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_val)

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="weighted")
    recall = recall_score(y_val, y_pred, average="weighted")
    f1 = f1_score(y_val, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_val, y_pred)

    # Print metrics
    print("Random Forest Classification (Custom) Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
