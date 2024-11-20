import pandas as pd
import numpy as np

# File paths
input_file = r"C:\Users\91985\drug-discovery-analysis\deep_learning\coordinates_copy.csv"
output_file = r"C:\Users\91985\drug-discovery-analysis\deep_learning\ligand_descriptors.csv"

# Function to calculate basic descriptors from coordinates
def calculate_descriptors_from_coordinates(df):
    data = []
    
    if 'PDB_ID' not in df.columns:
        print("Error: 'PDB_ID' column not found in the data.")
        return data
    
    grouped = df.groupby('PDB_ID')
    
    for pdb_id, group in grouped:
        coordinates = group[['X', 'Y', 'Z']].values
        
        if len(coordinates) == 0:
            print(f"Warning: No coordinates for PDB_ID {pdb_id}")
            continue
        
        # Compute descriptors
        centroid = np.mean(coordinates, axis=0)
        variance = np.var(coordinates, axis=0)
        stddev = np.std(coordinates, axis=0)
        num_atoms = len(coordinates)
        
        descriptors = {
            'PDB_ID': pdb_id,
            'NumAtoms': num_atoms,
            'Centroid_X': centroid[0],
            'Centroid_Y': centroid[1],
            'Centroid_Z': centroid[2],
            'Variance_X': variance[0],
            'Variance_Y': variance[1],
            'Variance_Z': variance[2],
            'StdDev_X': stddev[0],
            'StdDev_Y': stddev[1],
            'StdDev_Z': stddev[2]
        }
        
        data.append(descriptors)
    
    return data

# Read the coordinates from CSV
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    df = pd.DataFrame()  # Initialize an empty DataFrame if there's an error

# Print unique PDB_IDs and their counts
print("Unique PDB_IDs and their counts:")
print(df['PDB_ID'].value_counts())

# Compute descriptors
descriptors_data = calculate_descriptors_from_coordinates(df)

# Create a DataFrame
if descriptors_data:
    df_descriptors = pd.DataFrame(descriptors_data)
    
    # Print number of descriptors generated
    print(f"Number of descriptors generated: {len(df_descriptors)}")
    
    # Save to CSV
    try:
        df_descriptors.to_csv(output_file, index=False)
        print("Descriptors have been saved to:", output_file)
    except Exception as e:
        print(f"Error saving CSV file: {e}")
else:
    print("No descriptors to save.")
