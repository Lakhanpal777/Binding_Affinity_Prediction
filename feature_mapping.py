import torch
import pandas as pd
import numpy as np

# Load the concatenated output (previously saved tensor)
concatenated_output = torch.load('final_concatenated_output.pt')

# Ensure the tensor is detached from the computation graph before converting to NumPy
concatenated_output = concatenated_output.detach()

# Convert the concatenated output to a NumPy array for easier processing
data_array = concatenated_output.numpy()

# Define feature mapping based on the ranges you provided (ensuring accuracy for each layer)
def generate_feature_mapping(graph_dim, descriptor_dim, coordinate_dim):
    mapping = {}
    
    # Graph features (Layer 1) - Features 1 to 6
    for i in range(graph_dim):
        mapping[f"Graph_{i + 1}"] = i
    
    # Descriptor features (Layer 2) - Features 7 to 22
    for i in range(descriptor_dim):
        mapping[f"Descriptor_{i + 1}"] = graph_dim + i

    # Coordinate features (Layer 3) - Features 23 to 150
    for i in range(coordinate_dim):
        mapping[f"Coordinate_{i + 1}"] = graph_dim + descriptor_dim + i

    return mapping

# Correct dimensions of each layer based on the provided ranges
graph_dim = 6  # Graph layer features 1-6
descriptor_dim = 16  # Descriptor layer features 7-22
coordinate_dim = 128  # Coordinate layer features 23-150

# Generate the feature mapping with correct range
feature_mapping = generate_feature_mapping(graph_dim, descriptor_dim, coordinate_dim)

# Outlier detection across all features using IQR method
def detect_outliers(data, threshold=1.5):
    outliers = {}
    
    # Check each feature and detect outliers
    for feature_name, idx in feature_mapping.items():
        feature_data = data[:, idx]
        
        # Calculate IQR for outlier detection
        q1 = np.percentile(feature_data, 25)
        q3 = np.percentile(feature_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Identify outliers
        outlier_indices = np.where((feature_data < lower_bound) | (feature_data > upper_bound))[0]
        
        # Only record outliers if any are found
        if len(outlier_indices) > 0:
            outliers[feature_name] = outlier_indices.tolist()
    
    return outliers

# Detect outliers using a threshold of 1.5 (IQR method)
outlier_dict = detect_outliers(data_array, threshold=1.5)

# Summarize the outliers detected
summary = {
    "Feature": [],
    "Layer": [],
    "Outlier_Count": []
}

# Split features into Graph, Descriptor, and Coordinate layers based on the feature names
for feature, indices in outlier_dict.items():
    if feature.startswith("Graph"):
        layer = "Graph"
    elif feature.startswith("Descriptor"):
        layer = "Descriptor"
    elif feature.startswith("Coordinate"):
        layer = "Coordinate"
    else:
        layer = "Unknown"  # In case something unexpected happens
    
    summary["Feature"].append(feature)
    summary["Layer"].append(layer)
    summary["Outlier_Count"].append(len(indices))

# Convert the summary into a DataFrame for easy viewing
outlier_summary_df = pd.DataFrame(summary)

# Save the summary to a CSV file
outlier_summary_df.to_csv("outlier_summary_updated.csv", index=False)
print("Outlier summary saved to 'outlier_summary_updated.csv'")

# Checking total outliers across all layers
total_outliers = sum([len(indices) for indices in outlier_dict.values()])
print(f"Total Outliers Detected: {total_outliers}")
