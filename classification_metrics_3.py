import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from classification_tree_custom_3 import ClassificationTreeCustom

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

# Calculate confusion matrix and metrics
conf_matrix = confusion_matrix(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average="weighted")
recall = recall_score(y_val, y_pred, average="weighted")
f1 = f1_score(y_val, y_pred, average="weighted")

# Print metrics
print("Validation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Print confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)
