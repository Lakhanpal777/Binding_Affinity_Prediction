import os
import glob
import numpy as np
import torch
import csv
import joblib  # For loading the scaler
from Bio.PDB import PDBParser
from rdkit import Chem
import tensorflow as tf

# 🛠 Load the trained model
model = tf.keras.models.load_model("best_model_tuned.keras")

# 🛠 Load the corrected scaler for inverse transformation
target_scaler = joblib.load("target_scaler_fixed.pkl")

# 🛠 Define dataset path
dataset_path = r"C:\Users\91985\drug-discovery-analysis\deep_learning\PDBBind2020\refined-set"

# 🛠 Load the unique PDB IDs (Only these will be processed)
unique_pdb_file = "unique_pdb_2020.txt"
with open(unique_pdb_file, "r") as f:
    unique_pdb_ids = set(line.strip() for line in f)

# Initialize PDB parser
parser = PDBParser(QUIET=True)

# 🔹 Function to extract ligand coordinates from .mol2 or .sdf file
def read_ligand_coordinates(ligand_file):
    coordinates = []
    try:
        if ligand_file.endswith('.mol2'):
            mol = Chem.MolFromMol2File(ligand_file)
        elif ligand_file.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(ligand_file)
            mol = next(suppl)
        else:
            raise ValueError("Unsupported ligand file format")

        if mol is None:
            raise ValueError("Failed to read ligand file")

        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coordinates.append((atom.GetSymbol(), np.array([pos.x, pos.y, pos.z], dtype=np.float32)))

    except Exception as e:
        print(f"⚠️ Error reading ligand file {ligand_file}: {e}")
        return None

    return coordinates

# 🔹 Function to extract pocket coordinates from .pdb file
def read_pocket_coordinates(pocket_file):
    coordinates = []
    try:
        structure = parser.get_structure('protein', pocket_file)
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coordinates.append((atom.get_name(), atom.get_vector().get_array()))
    except Exception as e:
        print(f"⚠️ Error reading pocket file {pocket_file}: {e}")
        return None

    return coordinates

# 🔹 Function to extract input features for prediction
def extract_features(ligand_coords, pocket_coords):
    if ligand_coords is None or pocket_coords is None:
        return None  # Skip if missing files

    ligand_coords = np.array([coord[1] for coord in ligand_coords])  # Extract numerical coordinates
    pocket_coords = np.array([coord[1] for coord in pocket_coords])

    # Compute feature statistics (mean and standard deviation)
    ligand_mean = np.mean(ligand_coords, axis=0) if len(ligand_coords) > 0 else np.zeros(3)
    ligand_std = np.std(ligand_coords, axis=0) if len(ligand_coords) > 0 else np.zeros(3)

    pocket_mean = np.mean(pocket_coords, axis=0) if len(pocket_coords) > 0 else np.zeros(3)
    pocket_std = np.std(pocket_coords, axis=0) if len(pocket_coords) > 0 else np.zeros(3)

    # 🔹 Ensure input is 150-dimensional
    input_features = np.zeros((1, 150))
    input_features[0, :3] = ligand_mean
    input_features[0, 3:6] = ligand_std
    input_features[0, 6:9] = pocket_mean
    input_features[0, 9:12] = pocket_std

    return input_features

# 🔹 Function to predict binding affinity
def predict_binding_affinity(ligand_coords, pocket_coords):
    input_features = extract_features(ligand_coords, pocket_coords)
    if input_features is None:
        return None

    # Make prediction
    prediction_scaled = model.predict(input_features)[0][0]

    # Convert prediction back to original binding affinity range
    prediction = target_scaler.inverse_transform(np.array([[prediction_scaled]]))[0][0]

    return prediction

# 🔹 Iterate over only the unique PDB files and make predictions
results = []
pdb_folders = glob.glob(os.path.join(dataset_path, '*'))  # Get all PDB ID folders

for folder in pdb_folders:
    pdb_id = os.path.basename(folder)

    # ⚠️ Skip PDB IDs that are NOT in the unique list
    if pdb_id not in unique_pdb_ids:
        continue

    # Locate ligand and pocket files
    ligand_files = glob.glob(os.path.join(folder, '*_ligand.mol2')) + glob.glob(os.path.join(folder, '*_ligand.sdf'))
    pocket_file = os.path.join(folder, f'{pdb_id}_pocket.pdb')

    if not ligand_files or not os.path.exists(pocket_file):
        print(f"⚠️ Skipping {pdb_id} (Missing ligand or pocket file)")
        continue

    ligand_file = ligand_files[0]  # Use first found ligand file

    # Read coordinates
    ligand_coords = read_ligand_coordinates(ligand_file)
    pocket_coords = read_pocket_coordinates(pocket_file)

    if ligand_coords is None or pocket_coords is None:
        print(f"⚠️ Skipping {pdb_id} (Error reading files)")
        continue

    # Make prediction
    predicted_affinity = predict_binding_affinity(ligand_coords, pocket_coords)

    if predicted_affinity is not None:
        results.append((pdb_id, float(predicted_affinity)))  # Ensure affinity is a float
        print(f"✅ Predicted affinity for {pdb_id}: {predicted_affinity:.4f}")

# 🔹 Save predictions to CSV properly
output_file = "binding_affinity_predictions.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PDB_ID", "Predicted_Affinity"])
    writer.writerows(results)

print(f"\n📂 Predictions saved to: {output_file}")
