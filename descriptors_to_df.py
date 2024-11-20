import pandas as pd

# Use forward slashes in the file path
csv_file_path = r'C:\Users\91985\drug-discovery-analysis\deep_learning\final_descriptors_with_missing.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Display the first few rows to confirm it's loaded correctly
print(df.head())
