import torch
import torch.nn as nn
import pandas as pd

# Load 2D descriptors
file_path = r'C:\Users\91985\drug-discovery-analysis\deep_learning\cleaned_descriptors.csv'
data = pd.read_csv(file_path)

# Drop non-numeric columns and prepare inputs
pdb_ids = data['PDB_ID']
smiles = data['SMILES']  # Optional: keep for reference
numeric_columns = data.drop(columns=['PDB_ID', 'SMILES']).astype(float)

# Convert to PyTorch tensor
descriptors = torch.tensor(numeric_columns.values, dtype=torch.float32)

# Define Dense Neural Network
class DenseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DenseNetwork, self).__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))  # Final output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Network configuration
input_dim = descriptors.shape[1]  # Number of 2D descriptors
hidden_dims = [64, 32]  # Hidden layer sizes
output_dim = 16  # Output size of the dense network

# Instantiate the network
dense_network = DenseNetwork(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)

# Pass descriptors through the dense network
latent_features = dense_network(descriptors)

# Save latent features
torch.save(latent_features, 'dense_layer_output.pt')
print(f"Dense layer output saved to dense_layer_output.pt with shape {latent_features.shape}")

# Load the dense layer output
output = torch.load('dense_layer_output.pt')

# Display the shape and a snippet of the tensor
print("Output Shape:", output.shape)
print("Sample Data:\n", output[:5])  # Display the first 5 rows