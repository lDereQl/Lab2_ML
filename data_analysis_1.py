import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Create a directory for intermediate results if not already existing
os.makedirs("mid_data", exist_ok=True)

# Load the original dataset
data = pd.read_csv("hour.csv")  # Assuming the original dataset is named `hour.csv`

### Step 1: Data Splitting ###
# Define split sizes
train_size = 0.7
val_size = 0.2
test_size = 0.1

# Calculate validation split size relative to the train+validation set
train_val_size = 1 - test_size
val_relative_size = val_size / train_val_size

# Split the dataset into train+validation and test
train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

# Further split train+validation into train and validation
train_data, val_data = train_test_split(train_val_data, test_size=val_relative_size, random_state=42)

# Save the split datasets
train_data.to_csv("mid_data/train_data.csv", index=False)
val_data.to_csv("mid_data/val_data.csv", index=False)
test_data.to_csv("mid_data/test_data.csv", index=False)


# Step 2: Data Cleaning
def remove_outliers(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df


# Load datasets
train_data = pd.read_csv("mid_data/train_data.csv")
val_data = pd.read_csv("mid_data/val_data.csv")
test_data = pd.read_csv("mid_data/test_data.csv")

# List of numerical features for outlier removal
numerical_features = ['temp', 'atemp', 'hum', 'windspeed']  # Adjusted for normalized variables

# Remove outliers for training data only
train_data_clean = remove_outliers(train_data, numerical_features)

# Fill missing values with column means for numeric columns
numeric_columns = train_data_clean.select_dtypes(include=[np.number]).columns
train_data_clean[numeric_columns] = train_data_clean[numeric_columns].fillna(train_data_clean[numeric_columns].mean())
val_data[numeric_columns] = val_data[numeric_columns].fillna(val_data[numeric_columns].mean())
test_data[numeric_columns] = test_data[numeric_columns].fillna(test_data[numeric_columns].mean())

# Drop uninformative or redundant features
columns_to_drop = ['instant', 'dteday', 'casual',
                   'registered']  # Excluding `casual` and `registered` for target leakage prevention
train_data_clean.drop(columns=columns_to_drop, inplace=True, errors='ignore')
val_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Save cleaned datasets
train_data_clean.to_csv("mid_data/train_data_clean.csv", index=False)
val_data.to_csv("mid_data/val_data_clean.csv", index=False)
test_data.to_csv("mid_data/test_data_clean.csv", index=False)


### Step 3: Encoding ###
def encode_temporal_features(df):
    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
    df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
    return df


# Custom encoding for categorical features
def custom_one_hot_encoding(data):
    weather_dummies = pd.get_dummies(data['weathersit'], prefix='weathersit')
    if 'weathersit_4' in weather_dummies.columns:  # Drop the worst weather category
        weather_dummies.drop(columns=['weathersit_4'], inplace=True)
    data = pd.concat([data, weather_dummies], axis=1)
    data.drop(columns=['weathersit'], inplace=True)
    return data


# Apply encoding and feature engineering
train_data_clean = encode_temporal_features(train_data_clean)
val_data = encode_temporal_features(val_data)
test_data = encode_temporal_features(test_data)

train_data_encoded = custom_one_hot_encoding(train_data_clean)
val_data_encoded = custom_one_hot_encoding(val_data)
test_data_encoded = custom_one_hot_encoding(test_data)

# One-hot encode additional categorical features
categorical_features = ['season']  # Additional categorical features
train_data_encoded = pd.get_dummies(train_data_encoded, columns=categorical_features, drop_first=True)
val_data_encoded = pd.get_dummies(val_data_encoded, columns=categorical_features, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_encoded, columns=categorical_features, drop_first=True)

# Save final encoded datasets
train_data_encoded.to_csv("train_data_final.csv", index=False)
val_data_encoded.to_csv("val_data_final.csv", index=False)
test_data_encoded.to_csv("test_data_final.csv", index=False)

print(
    "Scripts applied successfully. Final datasets saved: 'train_data_final.csv', 'val_data_final.csv', 'test_data_final.csv'.")
