import os
import pandas as pd
from rdkit import Chem
import numpy as np

def read_smiles_file(file_path):
    """Reads SMILES and PDB IDs from a text file."""
    smiles_list = []
    pdb_ids = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    print("Skipping empty line.")
                    continue
                
                try:
                    smiles, pdb_id = line.split()
                    smiles_list.append(smiles)
                    pdb_ids.append(pdb_id)
                except ValueError:
                    print(f"Skipping line due to format issue: {line}")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    
    print(f"Read {len(smiles_list)} SMILES and PDB IDs from the file.")
    return smiles_list, pdb_ids

def smiles_to_graph(smiles):
    """Converts a SMILES string to atom features, adjacency list, and bond features."""
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print(f"Failed to process SMILES: {smiles}")
        return None

    # Adjusting atom features to a numpy array
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),             # Atomic number
            atom.GetTotalDegree(),           # Degree
            atom.GetFormalCharge(),           # Formal charge
            atom.GetTotalNumHs(),            # Number of hydrogens
            atom.GetIsAromatic(),            # Aromaticity
            atom.GetHybridization().real     # Hybridization type as integer (real to avoid issues)
        ])

    # Example adjacency list extraction
    adjacency_list = []
    bond_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency_list.append((i, j))

        # Bond type features (adjust as necessary)
        bond_type = [
            bond.GetBondTypeAsDouble() == 1.0,  # Single bond
            bond.GetBondTypeAsDouble() == 2.0,  # Double bond
            bond.GetBondTypeAsDouble() == 3.0,  # Triple bond
            bond.GetIsAromatic()                # Aromatic bond
        ]
        bond_features.append(bond_type)

    print(f"Processed SMILES: {smiles} into graph with {len(atom_features)} atoms and {len(adjacency_list)} bonds.")
    return atom_features, adjacency_list, bond_features

def process_ligands(smiles_file):
    """Processes ligands and creates a dictionary of graph features."""
    ligand_graphs = {}
    smiles_list, pdb_ids = read_smiles_file(smiles_file)
    
    for smiles, pdb_id in zip(smiles_list, pdb_ids):
        graph = smiles_to_graph(smiles)
        
        if graph is None:
            print(f"Skipping {pdb_id} due to processing failure.")
            continue  # Skip this molecule if it couldn't be processed

        atom_features, adjacency_list, bond_features = graph
        
        # Validate atom features
        if not atom_features:  # Check if atom features are empty
            print(f"Skipping {pdb_id} due to empty atom features.")
            continue
        
        ligand_graphs[pdb_id] = {
            "smiles": smiles,
            "atom_features": atom_features,
            "adjacency_list": adjacency_list,
            "bond_features": bond_features,
        }
    
    print(f"Processed {len(ligand_graphs)} ligands successfully.")
    return ligand_graphs

def save_to_csv(ligand_graphs, output_file):
    """Saves the ligand graph data to a CSV file."""
    data = {
        'PDB_ID': [],
        'SMILES': [],
        'Atom_Features': [],
        'Adjacency_List': [],
        'Bond_Features': []
    }
    
    for pdb_id, graph in ligand_graphs.items():
        data['PDB_ID'].append(pdb_id)
        data['SMILES'].append(graph['smiles'])
        data['Atom_Features'].append(graph['atom_features'])
        data['Adjacency_List'].append(graph['adjacency_list'])
        data['Bond_Features'].append(graph['bond_features'])
    
    df = pd.DataFrame(data)
    
    # Ensure all columns are appropriately formatted
    df['Atom_Features'] = df['Atom_Features'].apply(lambda x: np.array(x, dtype=object) if isinstance(x, list) else [])
    df['Adjacency_List'] = df['Adjacency_List'].apply(lambda x: np.array(x, dtype=object) if isinstance(x, list) else [])
    df['Bond_Features'] = df['Bond_Features'].apply(lambda x: np.array(x, dtype=object) if isinstance(x, list) else [])
    
    df.to_csv(output_file, index=False)
    print(f"Saved ligand graph data to {output_file}.")

if __name__ == "__main__":
    smiles_file = "ligand_smiles.txt"  # Path to the ligand SMILES file
    output_file = "ligand_graphs.csv"   # Output CSV file path
    
    # Check if the file exists
    if not os.path.exists(smiles_file):
        print(f"File {smiles_file} not found.")
    else:
        print(f"Starting processing of ligands from {smiles_file}.")
        # Process the ligands
        ligand_graphs = process_ligands(smiles_file)
        
        # Check if any ligands were processed
        if not ligand_graphs:
            print("No ligands were processed successfully.")
        else:
            # Save the results to a CSV file
            save_to_csv(ligand_graphs, output_file)

            # Optional: print first 5 ligands to confirm successful processing
            for pdb_id, graph in list(ligand_graphs.items())[:5]:
                print(f"PDB ID: {pdb_id}")
                print(f"SMILES: {graph['smiles']}")
                print(f"Atom Features: {graph['atom_features']}")
                print(f"Adjacency List: {graph['adjacency_list']}")
                print(f"Bond Features: {graph['bond_features']}\n")
