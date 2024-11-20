import pandas as pd
import numpy as np

# Step 1: Load coordinates file
coordinates_file = "coordinates.csv"  # Replace with actual file path
coordinates_df = pd.read_csv(coordinates_file, sep='\t')  # Assuming the file is tab-separated

# Step 2: Group coordinates by PDB_ID
# Each group will represent one molecule (PDB_ID), with all atom coordinates
grouped_coords = coordinates_df.groupby("PDB_ID")

# Function to calculate Radius of Gyration
def calc_radius_of_gyration(coords):
    center_of_mass = np.mean(coords, axis=0)
    diff = coords - center_of_mass
    radius_of_gyration = np.sqrt(np.sum(np.square(diff)) / len(coords))
    return radius_of_gyration

# Function to calculate Molecular Volume using bounding box
def calc_molecular_volume(coords):
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    volume = np.prod(max_coords - min_coords)
    return volume

# Step 3: Initialize list to store results
descriptor_results = []

# Step 4: Loop through each molecule (grouped by PDB_ID)
for pdb_id, group in grouped_coords:
    # Extract the X, Y, Z coordinates
    coords = group[['X', 'Y', 'Z']].values
    
    # Calculate 3D descriptors
    radius_of_gyration = calc_radius_of_gyration(coords)
    molecular_volume = calc_molecular_volume(coords)
    
    # Store the results in a dictionary
    descriptor_results.append({
        "PDB_ID": pdb_id,
        "Radius_of_Gyration": radius_of_gyration,
        "Molecular_Volume": molecular_volume
    })

# Step 5: Convert results to DataFrame
descriptors_df = pd.DataFrame(descriptor_results)

# Step 6: Save the new descriptors to a CSV file
descriptors_df.to_csv("3D_descriptors.csv", index=False)

# Step 7: Merge with the existing descriptor set (assuming existing descriptors are in 'existing_descriptors.csv')
existing_descriptors_df = pd.read_csv("existing_descriptors.csv")
final_df = pd.merge(existing_descriptors_df, descriptors_df, on="PDB_ID", how="inner")

# Save final descriptor set
final_df.to_csv("final_descriptors_with_3D.csv", index=False)

print("3D descriptors added and saved successfully.")
