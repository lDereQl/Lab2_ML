import numpy as np
import pandas as pd

class ClassificationTreeCustom:
    def __init__(self, max_depth=10, min_samples_split=20, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion  # "gini" or "entropy"
        self.tree = None

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Avoid log(0)

    def _criterion_score(self, y):
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _split(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] < threshold
        right_indices = ~left_indices
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature, best_threshold, best_score = None, None, float("inf")

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                score = (
                    len(y_left) * self._criterion_score(y_left)
                    + len(y_right) * self._criterion_score(y_right)
                ) / n_samples

                if score < best_score:
                    best_feature, best_threshold, best_score = feature_index, threshold, score

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or self._criterion_score(y) == 0:
            return np.argmax(np.bincount(y))

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.argmax(np.bincount(y))

        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return {"feature_index": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, sample, tree):
        while isinstance(tree, dict):
            if sample[tree["feature_index"]] < tree["threshold"]:
                tree = tree["left"]
            else:
                tree = tree["right"]
        return tree

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])


# Test the classification tree
if __name__ == "__main__":
    # Load dataset
    train_data = pd.read_csv("train_data_with_classes.csv")
    val_data = pd.read_csv("val_data_with_classes.csv")

    X_train = train_data.drop(columns=["cnt", "cnt_class"]).values
    y_train = train_data["cnt_class"].values.astype(int)

    X_val = val_data.drop(columns=["cnt", "cnt_class"]).values
    y_val = val_data["cnt_class"].values.astype(int)

    # Train and test the classification tree
    tree = ClassificationTreeCustom(max_depth=10, min_samples_split=20, criterion="gini")
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_val)

    # Calculate metrics
    accuracy = np.mean(y_pred == y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")
