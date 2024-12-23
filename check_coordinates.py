import pandas as pd

# Load the CSV file
coordinates_file = "coordinates_updated.csv"
df_coordinates = pd.read_csv(coordinates_file)

# Function to truncate lists to a fixed length
def truncate_lists(column, length):
    return df_coordinates[column].apply(lambda x: eval(x)[:length])

# Truncate ligand coordinates to 22 elements and pocket coordinates to 24 elements
df_coordinates['Ligand_X_Normalized'] = truncate_lists('Ligand_X_Normalized', 22)
df_coordinates['Ligand_Y_Normalized'] = truncate_lists('Ligand_Y_Normalized', 22)
df_coordinates['Ligand_Z_Normalized'] = truncate_lists('Ligand_Z_Normalized', 22)

df_coordinates['Pocket_X_Normalized'] = truncate_lists('Pocket_X_Normalized', 24)
df_coordinates['Pocket_Y_Normalized'] = truncate_lists('Pocket_Y_Normalized', 24)
df_coordinates['Pocket_Z_Normalized'] = truncate_lists('Pocket_Z_Normalized', 24)

# Save the truncated DataFrame to a new CSV file
df_coordinates.to_csv("coordinates_truncated.csv", index=False)
print("Truncated data saved to 'coordinates_truncated.csv'.")
