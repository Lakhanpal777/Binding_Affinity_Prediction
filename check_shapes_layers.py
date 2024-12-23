import torch
import h5py
import os

# Folder containing the input files
input_folder = r'C:\Users\91985\drug-discovery-analysis\deep_learning' # Update this path if needed

# File names
gnn_output_file = os.path.join(input_folder, "graph_layer_output.pt")
pooled_features_file = os.path.join(input_folder, "pooled_features.pt")
dense_output_file = os.path.join(input_folder, "dense_layer_output.pt")
coordinates_output_file = os.path.join(input_folder, "coordinate_model.h5")

# Load and print shapes from PyTorch files
def load_pt_file(file_path, file_description):
    try:
        data = torch.load(file_path)
        if isinstance(data, dict):
            print(f"{file_description} Contents:")
            for key, value in data.items():
                if hasattr(value, "shape"):
                    print(f"  - {key}: {value.shape}")
                else:
                    print(f"  - {key}: Non-tensor object (Type: {type(value)})")
        else:
            print(f"{file_description} Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {file_description}: {e}")
        return None

# Load and print shapes from HDF5 file
def load_h5_file(file_path, file_description):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"{file_description} Contents:")
            for key in f.keys():
                print(f"  - {key}: {f[key].shape}")
            return f[key]  # Return the first dataset for further use
    except Exception as e:
        print(f"Error loading {file_description}: {e}")
        return None

# Inspecting input files
print("Inspecting input files:\n")
gnn_output = load_pt_file(gnn_output_file, "GNN Layer Output")
pooled_output = load_pt_file(pooled_features_file, "Pooled Features Output")
dense_output = load_pt_file(dense_output_file, "Dense Layer Output")
coordinates_output = load_h5_file(coordinates_output_file, "Coordinates Layer Output")

# Extract necessary tensors for concatenation
if gnn_output is not None and 'node_features' in gnn_output:
    gnn_features = gnn_output['node_features']
    print(f"GNN Features Shape: {gnn_features.shape}")

if pooled_output is not None and 'pooled_features' in pooled_output:
    pooled_features = pooled_output['pooled_features']
    print(f"Pooled Features Shape: {pooled_features.shape}")

if isinstance(dense_output, torch.Tensor):
    print(f"Dense Layer Features Shape: {dense_output.shape}")
else:
    print("Dense Layer Output not found.")

# Extract coordinates (assuming a dataset 'coordinates' exists in the HDF5 file)
if coordinates_output is not None:
    coordinates_data = coordinates_output['coordinates'][:]
    print(f"Coordinates Data Shape: {coordinates_data.shape}")
else:
    print("Coordinates Output not found.")
