import pandas as pd
import re

# File path
file_path = r"C:\Users\91985\drug-discovery-analysis\deep_learning\output.txt"

# Function to parse coordinates
def parse_coordinates(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Processing PDB ID" in line:
                pdb_id = line.strip().split(": ")[1]
                print(f"Processing PDB ID: {pdb_id}")  # Debug print
            elif "Ligand Coordinates" in line:
                coordinates_str = line.strip().split(": ", 1)[1]
                # Extract tuples of coordinates from the array() format
                coordinates = re.findall(r"\('(\w+)', array\(\[([-\d.e ]+), ([-\d.e ]+), ([-\d.e ]+)\], dtype=float32\)\)", coordinates_str)
                if not coordinates:
                    print(f"Failed to extract coordinates from line: {line.strip()}")  # Debug print
                for atom in coordinates:
                    atom_type, x, y, z = atom
                    data.append([pdb_id, atom_type, float(x), float(y), float(z)])
    return data

# Parse the coordinates and create a DataFrame
data = parse_coordinates(file_path)
df = pd.DataFrame(data, columns=['PDB_ID', 'Atom_Type', 'X', 'Y', 'Z'])

# Display the DataFrame
print(df.head())

# Save the DataFrame to a CSV file (optional)
df.to_csv(r"C:\Users\91985\drug-discovery-analysis\deep_learning\coordinates.csv", index=False)
