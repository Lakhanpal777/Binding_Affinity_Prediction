import pandas as pd
import numpy as np
import json

# Load the data from an Excel file
df = pd.read_excel(r'C:\Users\91985\drug-discovery-analysis\src\output_dataframed.xlsx')

# Print the first few rows of the original data
print("Original Data Sample:", df.head())

# Function to convert to valid JSON format
def convert_to_valid_json(s):
    try:
        # Replace single quotes with double quotes for JSON compatibility
        valid_json = s.replace("'", '"')
        # Attempt to parse the JSON string
        return json.loads(valid_json)
    except json.JSONDecodeError:
        print(f"Invalid JSON: {s}")
        return []

# Apply the JSON conversion to the ligand and pocket coordinates
df['Ligand_Coordinates'] = df['Ligand_Coordinates'].apply(convert_to_valid_json)
df['Pocket_Coordinates'] = df['Pocket_Coordinates'].apply(convert_to_valid_json)

# Debug: Print first few rows of the parsed coordinates
print("Parsed Ligand Coordinates Sample:", df['Ligand_Coordinates'].head())
print("Parsed Pocket Coordinates Sample:", df['Pocket_Coordinates'].head())

# Function to extract basic statistics from coordinates
def extract_features(coord_list):
    if len(coord_list) == 0:
        return [0, 0, 0, 0, 0, 0, 0]  # Return zeros if the list is empty
    
    # Extract only the coordinates from the nested list
    coords = np.array([coord[1] for coord in coord_list])
    
    return [
        len(coords),  # Number of coordinates
        np.mean(coords[:, 0]),  # Mean X
        np.mean(coords[:, 1]),  # Mean Y
        np.mean(coords[:, 2]),  # Mean Z
        np.std(coords[:, 0]),  # Std X
        np.std(coords[:, 1]),  # Std Y
        np.std(coords[:, 2])   # Std Z
    ]

# Apply the feature extraction to the ligand and pocket coordinates
df['Ligand_Features'] = df['Ligand_Coordinates'].apply(extract_features)
df['Pocket_Features'] = df['Pocket_Coordinates'].apply(extract_features)

# Debug: Print first few rows of the extracted features
print("Extracted Ligand Features Sample:", df['Ligand_Features'].head())
print("Extracted Pocket Features Sample:", df['Pocket_Features'].head())

# Convert lists of features into DataFrames
ligand_features = pd.DataFrame(df['Ligand_Features'].tolist(), columns=['Num_Ligand_Coords', 'Ligand_Mean_X', 'Ligand_Mean_Y', 'Ligand_Mean_Z', 'Ligand_Std_X', 'Ligand_Std_Y', 'Ligand_Std_Z'])
pocket_features = pd.DataFrame(df['Pocket_Features'].tolist(), columns=['Num_Pocket_Coords', 'Pocket_Mean_X', 'Pocket_Mean_Y', 'Pocket_Mean_Z', 'Pocket_Std_X', 'Pocket_Std_Y', 'Pocket_Std_Z'])

# Combine features into a single DataFrame
features = pd.concat([ligand_features, pocket_features], axis=1)

# Debug: Print first few rows of the combined features
print("Combined Features Sample:", features.head())

# Save the features to an Excel file
features.to_excel(r'C:\Users\91985\drug-discovery-analysis\src\features_debug.xlsx', index=False)
