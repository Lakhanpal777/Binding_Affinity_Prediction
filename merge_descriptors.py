import pandas as pd

# Load the PDB-SMILES mapping file
pdb_smiles = pd.read_csv('pdb_smiles.csv')

# Load the 2D descriptors file (which currently only has SMILES, not PDB_ID)
descriptors_2d = pd.read_csv('descriptors.csv')

# Merge the SMILES with PDB_ID from the pdb_smiles.csv file
descriptors_with_pdb = pd.merge(descriptors_2d, pdb_smiles, on='SMILES', how='inner')

# Save this file with PDB_ID included for future use
descriptors_with_pdb.to_csv('descriptors_with_pdb.csv', index=False)

# Load the 3D descriptors file
descriptors_3d = pd.read_csv('3D_descriptors.csv')

# Now merge the 2D descriptors (with PDB_ID) and the 3D descriptors on PDB_ID
final_descriptors = pd.merge(descriptors_with_pdb, descriptors_3d, on='PDB_ID', how='inner')

# Save the final combined descriptors to a CSV file
final_descriptors.to_csv('final_descriptors.csv', index=False)

print("Final descriptors file saved as final_descriptors.csv.")
