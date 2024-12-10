# Load required libraries
import pandas as pd
from decision_tree_custom import DecisionTreeRegressorCustom, calculate_mae, calculate_mse, calculate_mape
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Load the training dataset
train_data_path = "train_data_final.csv"  # Ensure this file is available
train_data = pd.read_csv(train_data_path)
X = train_data.drop(columns=["cnt"]).values
y = train_data["cnt"].values

# Load the test dataset
test_data_path = "test_data_final.csv"  # Ensure this file is available
test_data = pd.read_csv(test_data_path)
X_test = test_data.drop(columns=["cnt"]).values
y_test = test_data["cnt"].values

# Train the custom model with the best parameters
custom_model = DecisionTreeRegressorCustom(max_depth=15, min_samples_split=10)
custom_model.fit(X, y)  # Train on the full training set
y_pred_custom = custom_model.predict(X_test)

# Train the Scikit-learn model with the best parameters
sklearn_model = DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42)
sklearn_model.fit(X, y)  # Train on the full training set
y_pred_sklearn = sklearn_model.predict(X_test)

# Evaluate test set performance for both models
custom_metrics = {
    "Model": "Custom",
    "Test MSE": calculate_mse(y_test, y_pred_custom),
    "Test MAE": calculate_mae(y_test, y_pred_custom),
    "Test MAPE": calculate_mape(y_test, y_pred_custom),
}

sklearn_metrics = {
    "Model": "Scikit-learn",
    "Test MSE": calculate_mse(y_test, y_pred_sklearn),
    "Test MAE": calculate_mae(y_test, y_pred_sklearn),
    "Test MAPE": calculate_mape(y_test, y_pred_sklearn),
}

# Combine results
test_results = pd.DataFrame([custom_metrics, sklearn_metrics])

# Save and display test results
test_results.to_csv("test_set_evaluation_results.csv", index=False)
print("Test results saved to 'test_set_evaluation_results.csv'.")
# Load the evaluation results (if not already in memory)
test_results = pd.read_csv("test_set_evaluation_results.csv")

# Bar chart data
models = test_results["Model"].tolist()
mse = test_results["Test MSE"].tolist()
mae = test_results["Test MAE"].tolist()
mape = test_results["Test MAPE"].tolist()

metrics = ["MSE", "MAE", "MAPE"]
custom_scores = [mse[0], mae[0], mape[0]]
sklearn_scores = [mse[1], mae[1], mape[1]]

x = range(len(metrics))

# Create a combined plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Bar chart
ax1.bar(x, custom_scores, width=0.4, label="Custom", align="center")
ax1.bar([i + 0.4 for i in x], sklearn_scores, width=0.4, label="Scikit-learn", align="center")
ax1.set_xticks([i + 0.2 for i in x])
ax1.set_xticklabels(metrics)
ax1.set_ylabel("Metric Value")
ax1.set_title("Test Set Evaluation: Custom vs Scikit-learn Decision Trees")
ax1.legend()

# Table
ax2.axis("tight")
ax2.axis("off")
table = ax2.table(cellText=test_results.values, colLabels=test_results.columns, loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(test_results.columns))))

plt.tight_layout()
plt.show()