import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Function to read SMILES strings from the file
def read_smiles_file(filename):
    smiles_list = []
    with open(filename, 'r') as file:
        current_smiles = ""
        for line in file:
            if '\t' in line:
                if current_smiles:
                    smiles_list.append(current_smiles)
                current_smiles = line.strip()
            else:
                current_smiles += line.strip()
        if current_smiles:
            smiles_list.append(current_smiles)
    return [(smiles.split('\t')[0], smiles.split('\t')[1]) for smiles in smiles_list]

# Function to generate molecular descriptors
def generate_descriptors(smiles_list):
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calc = MolecularDescriptorCalculator(descriptor_names)
    morgan_generator = GetMorganGenerator(radius=2, fpSize=1024)  # Use GetMorganGenerator
    
    descriptor_data = []
    
    for smiles, pdb_id in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptors = calc.CalcDescriptors(mol)
            morgan_fp = morgan_generator.GetFingerprint(mol).ToBitString()
            morgan_fp_list = [int(bit) for bit in morgan_fp]
            descriptor_data.append([pdb_id] + list(descriptors) + morgan_fp_list)
    
    return descriptor_names + [f'MorganBit_{i}' for i in range(1024)], descriptor_data

# Main function to execute the code
if __name__ == "__main__":
    # Set the file paths
    smiles_file = "ligand_smiles.txt"
    output_file = "ligand_descriptors.xlsx"
    
    # Read SMILES strings and PDB IDs
    smiles_list = read_smiles_file(smiles_file)
    
    # Generate descriptors
    descriptor_names, descriptor_data = generate_descriptors(smiles_list)
    
    # Create a DataFrame and save to Excel
    df = pd.DataFrame(descriptor_data, columns=["PDB_ID"] + descriptor_names)
    df.to_excel(output_file, index=False)
    
    print(f"Descriptors generated and saved to {output_file}")
