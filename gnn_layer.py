import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd
import ast

# Load Data (adjust the path if necessary)
data = pd.read_csv(r'C:\Users\91985\drug-discovery-analysis\deep_learning\ligand_graphs_parsed.csv', dtype=str)

# Function to parse list fields safely
def parse_list_field(field_value):
    try:
        # Convert string representation of list to an actual list
        return ast.literal_eval(field_value)
    except (ValueError, SyntaxError):
        print(f"Warning: Could not parse {field_value} as a list")
        return []

# Parse complex fields
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
    else:
        print(f"Skipping row due to parsing error: {row['Parsed_Atom_Features']}, {row['Parsed_Adjacency_List']}")

# DataLoader for batching
loader = DataLoader(graph_data_list, batch_size=32, shuffle=True)

# Define the GNN Layer
class GCNLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply the first GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Apply activation

        # Apply the second GCN layer
        x = self.conv2(x, edge_index)
        return x

# Instantiate the GCN Layer
# Assuming atom features have 6 dimensions, hidden layer has 16, output layer has 8
gcn_layer = GCNLayer(input_dim=6, hidden_dim=16, output_dim=8)

# Process batches of data through the GCN layer
for batch in loader:
    batch_output = gcn_layer(batch)  # Output for one batch of graphs
    print("Batch processed. Output shape:", batch_output.shape)
