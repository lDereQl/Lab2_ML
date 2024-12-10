import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load datasets
train_data = pd.read_csv("train_data_with_classes.csv")
val_data = pd.read_csv("val_data_with_classes.csv")

# Extract feature names
features = train_data.drop(columns=["cnt", "cnt_class"]).columns

# Separate features and targets
X_train_class = train_data[features].values
y_train_class = train_data["cnt_class"].values.astype(int)

X_train_reg = train_data[features].values
y_train_reg = train_data["cnt"].values

# Train Random Forest Classifier for feature importance (classification)
rf_class = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_class.fit(X_train_class, y_train_class)

# Train Random Forest Regressor for feature importance (regression)
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

# Extract feature importances
class_importances = rf_class.feature_importances_
reg_importances = rf_reg.feature_importances_

# Plot feature importances for classification
plt.figure(figsize=(10, 6))
plt.barh(features, class_importances, align="center")
plt.title("Feature Importance (Classification)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()

# Plot feature importances for regression
plt.figure(figsize=(10, 6))
plt.barh(features, reg_importances, align="center")
plt.title("Feature Importance (Regression)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()
