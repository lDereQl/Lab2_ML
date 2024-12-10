from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Load dataset
train_data = pd.read_csv("train_data_with_classes.csv")
val_data = pd.read_csv("val_data_with_classes.csv")

X_train = train_data.drop(columns=["cnt", "cnt_class"]).values
y_train = train_data["cnt_class"].values.astype(int)

X_val = val_data.drop(columns=["cnt", "cnt_class"]).values
y_val = val_data["cnt_class"].values.astype(int)

# Train Scikit-learn Random Forest
rf = RandomForestClassifier(
    n_estimators=50, max_depth=10, random_state=42, max_features="sqrt"
)
rf.fit(X_train, y_train)

# Predict on validation data
y_pred = rf.predict(X_val)

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average="weighted")
recall = recall_score(y_val, y_pred, average="weighted")
f1 = f1_score(y_val, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_val, y_pred)

# Print metrics
print("Scikit-learn Random Forest Classification Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
