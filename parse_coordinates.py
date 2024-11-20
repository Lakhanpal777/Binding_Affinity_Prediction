import re
import pandas as pd

def extract_coordinates(data, coord_type):
    pattern = f"{coord_type}: \\[([^\\]]+)\\]"
    match = re.search(pattern, data)
    if match:
        coord_strings = match.group(1).split("),")
        coords = []
        for coord in coord_strings:
            coord = coord.strip().strip("()")
            parts = coord.split(", array(")
            if len(parts) == 2:
                atom, values = parts
                values = values.strip("[]").split(", ")
                values = list(map(float, values))
                coords.append((atom.strip("' "), values))
        return coords
    else:
        print(f"No match found for {coord_type}")
        return []

def process_file(input_file):
    data = []
    try:
        with open(input_file, 'r') as file:
            content = file.read()
            print("File content preview:")
            print(content[:1000])  # Print the first 1000 characters for debugging
            
            pdb_ids = re.findall(r'PDB ID: (\w+)', content)
            if not pdb_ids:
                print("No PDB IDs found in the file.")
            
            for pdb_id in pdb_ids:
                start_index = content.find(f'PDB ID: {pdb_id}')
                next_pdb_start_index = content.find('PDB ID:', start_index + 1)
                end_index = next_pdb_start_index if next_pdb_start_index != -1 else len(content)
                pdb_data = content[start_index:end_index]
                
                print(f"Processing PDB ID: {pdb_id}")
                print("PDB Data snippet:")
                print(pdb_data[:500])  # Print the first 500 characters of the PDB data for debugging
                
                ligand_coords = extract_coordinates(pdb_data, 'Ligand Coordinates')
                pocket_coords = extract_coordinates(pdb_data, 'Pocket Coordinates')
                
                data.append({
                    'PDB ID': pdb_id,
                    'Ligand Coordinates': ligand_coords,
                    'Pocket Coordinates': pocket_coords
                })
    except Exception as e:
        print(f"Error processing file: {e}")
    
    return data

def save_to_excel(data, output_file):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

# Main script
input_file = 'output.txt'
output_file = 'coordinates.xlsx'

data = process_file(input_file)
print("Processed data:")
print(data[:5])  # Print the first 5 entries for debugging

if data:
    save_to_excel(data, output_file)
else:
    print("No data to save.")
