import pandas as pd

# Sample: Replace these values with the actual indices of outliers for each feature
outlier_indices = {
    "Graph_1": [1, 2, 3, 5, 49],
    "Graph_2": [2, 5, 6, 10],
    "Graph_3": [3, 4, 7, 57, 88],
    "Graph_4": [49, 88, 100],
    "Graph_6": [57, 70, 90],
    "Descriptor_1": [4, 57, 88],
    "Descriptor_2": [1, 50],
    "Descriptor_3": [3, 4],
    "Descriptor_4": [49],
    "Descriptor_5": [57],
    "Descriptor_6": [2, 88],
    "Descriptor_7": [70, 90],
    "Descriptor_8": [50, 57],
    "Descriptor_9": [3, 5, 49],
    "Descriptor_10": [88],
    "Descriptor_11": [1],
    "Descriptor_12": [4],
    "Descriptor_13": [57],
    "Descriptor_14": [90],
    "Descriptor_15": [70],
    "Descriptor_16": [88],
}

# Step 1: Aggregate all outlier compound indices
all_outliers = set()
for feature, indices in outlier_indices.items():
    all_outliers.update(indices)

# Step 2: Calculate total unique compounds
total_unique_outliers = len(all_outliers)

# Output the results
print(f"Total unique compounds flagged as outliers: {total_unique_outliers}")
print(f"Unique compound indices: {sorted(all_outliers)}")

# Step 3: Save to a file for reference (optional)
outliers_df = pd.DataFrame({"Compound_ID": sorted(all_outliers)})
outliers_df.to_csv("unique_outliers.csv", index=False)
print("Outlier compound indices saved to 'unique_outliers.csv'")
