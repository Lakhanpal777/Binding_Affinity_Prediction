import torch
import pandas as pd

# Load the expanded pooled graph features, descriptors, and coordinates
graphs_output = torch.load('pooled_features_final.pt')['pooled_features']
descriptors_output = torch.load('dense_layer_output.pt')
coordinates_output = torch.load('dense_layer2_output.pt')

# Ensure all tensors have the same number of samples
assert graphs_output.shape[0] == descriptors_output.shape[0] == coordinates_output.shape[0], \
    f"Mismatch in the number of samples: " \
    f"graphs_output({graphs_output.shape[0]}), " \
    f"descriptors_output({descriptors_output.shape[0]}), " \
    f"coordinates_output({coordinates_output.shape[0]})"

# Concatenate the tensors along the feature dimension (dim=1)
concatenated_output = torch.cat([graphs_output, descriptors_output, coordinates_output], dim=1)

# Convert the concatenated tensor to a DataFrame
df = pd.DataFrame(concatenated_output.detach().numpy())

# Load the ligand SMILES file (ensure tab-separated)
ligand_smiles = pd.read_csv('ligand_smiles_corrected.csv')  # Corrected file name

# Strip any extra whitespace from column names
ligand_smiles.columns = ligand_smiles.columns.str.strip()

# Debugging: Verify column names and shape
print(f"Columns in ligand_smiles: {ligand_smiles.columns.tolist()}")
print(f"Shape of ligand_smiles: {ligand_smiles.shape}")
print(f"Shape of concatenated output: {df.shape}")

# Check if the number of rows in ligand_smiles matches the concatenated output
if ligand_smiles.shape[0] == df.shape[0]:
    # Add PDB_ID and SMILES to the DataFrame
    df.insert(0, 'PDB_ID', ligand_smiles['PDB_ID'])  # Insert PDB_ID as the first column
    df.insert(1, 'SMILES', ligand_smiles['SMILES'])  # Insert SMILES as the second column
else:
    print(f"Error: Number of rows in ligand_smiles ({ligand_smiles.shape[0]}) does not match the number of rows in concatenated output ({df.shape[0]}).")
    exit()

# Save the DataFrame to CSV
df.to_csv('concatenate_data_smilespdb.csv', index=False)
print("Final input saved to 'concatenate_data_smilespdb.csv'")
