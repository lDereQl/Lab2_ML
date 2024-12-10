import pandas as pd
import numpy as np

# Load the dataset
input_file = "train_data_final.csv"  # Input training dataset
output_file = "train_data_with_classes.csv"  # Output dataset with classes
val_output_file = "val_data_with_classes.csv"  # Validation set with classes

# Read the dataset
data = pd.read_csv(input_file)

# Define class boundaries (Example: Equal-frequency bins)
num_classes = 5
data["cnt_class"] = pd.qcut(data["cnt"], q=num_classes, labels=range(num_classes))

# Print class distribution
class_distribution = data["cnt_class"].value_counts()
print("Class Distribution:")
print(class_distribution)

# Save the updated dataset
data.to_csv(output_file, index=False)
print(f"Updated dataset with classes saved to {output_file}")

# Split Validation Data
validation_data_path = "val_data_final.csv"  # Assuming validation data is available
val_data = pd.read_csv(validation_data_path)
val_data["cnt_class"] = pd.qcut(val_data["cnt"], q=num_classes, labels=range(num_classes))
val_data.to_csv(val_output_file, index=False)
print(f"Validation dataset with classes saved to {val_output_file}")
