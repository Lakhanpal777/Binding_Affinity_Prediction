import torch
import torch_geometric.nn as geom_nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd
import ast

# Load and parse the CSV data
data = pd.read_csv(r'C:\Users\91985\drug-discovery-analysis\deep_learning\ligand_graphs_parsed.csv', dtype=str)

def parse_list_field(field_value):
    try:
        return ast.literal_eval(field_value)
    except (ValueError, SyntaxError):
        print(f"Warning: Could not parse {field_value} as a list")
        return []

data['Parsed_Atom_Features'] = data['Parsed_Atom_Features'].apply(parse_list_field)
data['Parsed_Adjacency_List'] = data['Parsed_Adjacency_List'].apply(parse_list_field)
data['Parsed_Bond_Features'] = data['Parsed_Bond_Features'].apply(parse_list_field)

# Convert to PyTorch Geometric Data objects
graph_data_list = []
for _, row in data.iterrows():
    if isinstance(row['Parsed_Atom_Features'], list) and isinstance(row['Parsed_Adjacency_List'], list):
        atom_features = torch.tensor(row['Parsed_Atom_Features'], dtype=torch.float)
        edge_index = torch.tensor(row['Parsed_Adjacency_List'], dtype=torch.long).t().contiguous()
        bond_features = torch.tensor(row['Parsed_Bond_Features'], dtype=torch.float)

        graph_data = Data(x=atom_features, edge_index=edge_index, edge_attr=bond_features)
        graph_data_list.append(graph_data)

# DataLoader for batching
loader = DataLoader(graph_data_list, batch_size=32, shuffle=False)  # Use shuffle=False for consistent ordering

# Pooling function
global_mean_pool = geom_nn.global_mean_pool

# Accumulate pooled features
pooled_features_list = []

# Process each batch and pool the features
for batch in loader:
    batch_pooled_features = global_mean_pool(batch.x, batch.batch)
    pooled_features_list.append(batch_pooled_features)

# Concatenate all pooled features from each batch to form a single tensor
pooled_features = torch.cat(pooled_features_list, dim=0)

# Check if the shape matches the required 4459 samples
assert pooled_features.shape[0] == 4459, f"Expected 4459 samples, but got {pooled_features.shape[0]}."

# Save the pooled features for further processing
pooled_output_path = 'pooled_features_final.pt'
torch.save({'pooled_features': pooled_features}, pooled_output_path)
print(f"Pooled features saved to '{pooled_output_path}' with shape {pooled_features.shape}")
