import torch
import pandas as pd
import numpy as np

# File paths
data_file = "final_concatenated_output.pt"  # Concatenated dataset
outliers_file = "unique_outliers.csv"      # File containing outlier indices
output_file = "filtered_data.pt"           # File to save filtered dataset

# Load the concatenated dataset
data = torch.load(data_file)  # Assuming this is a PyTorch tensor

# Load the outlier indices
outlier_indices = pd.read_csv(outliers_file)["Compound_ID"].tolist()

# Convert the data to a NumPy array for easier manipulation
data_np = data.detach().numpy() if data.requires_grad else data.numpy()

# Remove rows corresponding to outlier indices
filtered_data_np = np.delete(data_np, outlier_indices, axis=0)

# Convert back to a PyTorch tensor
filtered_data = torch.tensor(filtered_data_np)

# Save the filtered dataset
torch.save(filtered_data, output_file)

print(f"Filtered dataset saved to '{output_file}'.")
print(f"Total compounds removed: {len(outlier_indices)}")
print(f"Remaining compounds: {filtered_data.shape[0]}")
