import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

# Load the coordinates CSV
df = pd.read_csv('coordinates.csv')

# Function to calculate center of mass
def calculate_center_of_mass(coords):
    return np.mean(coords, axis=0)

# Function to calculate radius of gyration
def calculate_radius_of_gyration(coords, center_of_mass):
    squared_distances = np.sum((coords - center_of_mass) ** 2, axis=1)
    return np.sqrt(np.mean(squared_distances))

# Function to calculate molecular volume using convex hull
def calculate_molecular_volume(coords):
    try:
        hull = ConvexHull(coords)
        return hull.volume
    except:
        return np.nan  # Handle cases where volume can't be calculated

# Initialize list to store results
results = []

# Loop through each PDB_ID
for pdb_id in df['PDB_ID'].unique():
    pdb_df = df[df['PDB_ID'] == pdb_id]
    
    # Extract coordinates as a numpy array
    coords = pdb_df[['X', 'Y', 'Z']].values
    
    # Calculate center of mass
    center_of_mass = calculate_center_of_mass(coords)
    
    # Calculate radius of gyration
    radius_of_gyration = calculate_radius_of_gyration(coords, center_of_mass)
    
    # Calculate molecular volume
    molecular_volume = calculate_molecular_volume(coords)
    
    # Append results
    results.append({
        'PDB_ID': pdb_id,
        'RadiusOfGyration': radius_of_gyration,
        'MolecularVolume': molecular_volume
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('3D_descriptors.csv', index=False)

print("3D Descriptors calculation complete. Results saved to 3D_descriptors.csv.")
