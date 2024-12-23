import pandas as pd

# File paths
target_variable_file = "target_variable.txt"
feature_file = r'C:\Users\91985\drug-discovery-analysis\deep_learning\concatenate_data_smilespdb.csv'  

# Load the PDB IDs from your dataset
feature_data = pd.read_csv(feature_file)
valid_pdb_ids = set(feature_data["PDB_ID"])  # Extract unique PDB IDs from the dataset

# Initialize lists for filtered data
filtered_pdb_ids = []
filtered_affinities = []

# Open and process the target variable file
unmatched_pdb_ids = valid_pdb_ids.copy()  # Track unmatched PDB IDs
with open(target_variable_file, "r") as file:
    for line in file:
        parts = line.split()
        if len(parts) >= 5:
            pdb_id = parts[0]
            affinity_str = parts[4]
            
            # Check if the PDB ID is valid
            if pdb_id in valid_pdb_ids:
                unmatched_pdb_ids.discard(pdb_id)  # Remove matched ID from unmatched list
                
                # Parse and convert affinity values to nM
                if "Ki=" in affinity_str or "Kd=" in affinity_str or "IC50=" in affinity_str:
                    try:
                        value = float(affinity_str.split('=')[1][:-2])  # Remove 'mM'
                        binding_affinity_nM = value * 1_000_000  # Convert to nM
                        filtered_pdb_ids.append(pdb_id)
                        filtered_affinities.append(binding_affinity_nM)
                    except ValueError:
                        print(f"Error parsing affinity for {pdb_id}: {affinity_str}")
                else:
                    print(f"Unrecognized format for {pdb_id}: {affinity_str}")

# Log unmatched PDB IDs
if unmatched_pdb_ids:
    print(f"Unmatched PDB IDs: {unmatched_pdb_ids}")

# Create a DataFrame for filtered data
filtered_data = pd.DataFrame({
    "PDB_ID": filtered_pdb_ids,
    "Binding_Affinity": filtered_affinities
})

# Save the filtered data to a CSV file
output_file = "filtered_binding_affinity.csv"
filtered_data.to_csv(output_file, index=False)
print(f"Filtered binding affinity data saved to '{output_file}'.")

# Additional check for missing IDs
missing_ids = valid_pdb_ids.difference(filtered_data["PDB_ID"].unique())
if missing_ids:
    print(f"Missing PDB IDs in final data: {missing_ids}")
else:
    print("All valid PDB IDs were successfully processed.")
