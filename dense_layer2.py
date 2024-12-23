import pandas as pd
import torch
import torch.nn as nn

# Load the expanded coordinates from the CSV file
df_expanded = pd.read_csv("coordinates_expanded.csv")

# Drop non-numeric columns like 'PDB_ID', if present, and ensure all data is numeric
df_numeric = df_expanded.drop(columns=['PDB_ID'], errors='ignore').apply(pd.to_numeric, errors='coerce')

# Replace NaN values with 0 
df_numeric = df_numeric.fillna(0)

# Convert the DataFrame to a PyTorch tensor
input_data = torch.tensor(df_numeric.values, dtype=torch.float32)

# Define the dense layer (adjust output size as needed)
class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

# Initialize the dense layer with appropriate input and output sizes
input_size = input_data.shape[1]  # Number of features (columns)
output_size = 128  # Example output size (adjust this as needed)
dense_layer = DenseLayer(input_size, output_size)

# Forward pass: pass the input data through the dense layer
output_data = dense_layer(input_data)

# Save the output of the dense layer to a .pt file
torch.save(output_data, "dense_layer2_output.pt")

print("Dense layer output saved to 'dense_layer2_output.pt'")
