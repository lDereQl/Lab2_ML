import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create a directory for storing EDA results
os.makedirs("eda_outputs", exist_ok=True)

# Load cleaned datasets
train_data = pd.read_csv("mid_data/train_data_clean.csv")
val_data = pd.read_csv("mid_data/val_data_clean.csv")
test_data = pd.read_csv("mid_data/test_data_clean.csv")

### 1. Exploratory Data Analysis (EDA) ###
# Summary statistics
summary_stats = train_data.describe(include='all').transpose()
summary_stats.to_csv("eda_outputs/train_summary_stats.csv")
print("Summary statistics saved to 'eda_outputs/train_summary_stats.csv'.")

# Distribution plots for continuous features
continuous_features = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
for feature in continuous_features:
    plt.figure()
    sns.histplot(train_data[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.savefig(f"eda_outputs/{feature}_distribution.png")
    plt.close()

# Box plots to check for outliers
for feature in continuous_features:
    plt.figure()
    sns.boxplot(data=train_data, y=feature)
    plt.title(f'Boxplot of {feature}')
    plt.savefig(f"eda_outputs/{feature}_boxplot.png")
    plt.close()

# Bar plots for categorical features
categorical_features = ['season', 'weathersit', 'yr', 'mnth', 'hr', 'weekday']
for feature in categorical_features:
    plt.figure()
    sns.countplot(data=train_data, x=feature)
    plt.title(f'Counts of {feature}')
    plt.xticks(rotation=45)
    plt.savefig(f"eda_outputs/{feature}_countplot.png")
    plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.close()

### 2. Validation Checks ###
# Check for missing values
missing_values = pd.DataFrame({
    'train': train_data.isnull().sum(),
    'val': val_data.isnull().sum(),
    'test': test_data.isnull().sum()
})
missing_values.to_csv("eda_outputs/missing_values_check.csv")
print("Missing values check saved to 'eda_outputs/missing_values_check.csv'.")

# Check feature consistency across datasets
train_cols = set(train_data.columns)
val_cols = set(val_data.columns)
test_cols = set(test_data.columns)

missing_in_val = train_cols - val_cols
missing_in_test = train_cols - test_cols

with open("eda_outputs/feature_consistency_check.txt", "w") as f:
    f.write("Features missing in validation set:\n")
    f.write(", ".join(missing_in_val) + "\n")
    f.write("\nFeatures missing in test set:\n")
    f.write(", ".join(missing_in_test) + "\n")

print("Feature consistency check saved to 'eda_outputs/feature_consistency_check.txt'.")

### 3. Advanced Handling Suggestions ###
# Identify outliers using z-scores
outlier_threshold = 3
outliers = pd.DataFrame()
for feature in continuous_features:
    z_scores = np.abs((train_data[feature] - train_data[feature].mean()) / train_data[feature].std())
    outliers[feature] = (z_scores > outlier_threshold).sum()

outliers.to_csv("eda_outputs/outlier_detection.csv")
print("Outlier detection results saved to 'eda_outputs/outlier_detection.csv'.")

print("EDA outputs and additional checks completed. Results saved in 'eda_outputs' directory.")
