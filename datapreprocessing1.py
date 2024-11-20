import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a pandas DataFrame
csv_file_path = r'C:\Users\91985\drug-discovery-analysis\deep_learning\final_descriptors_with_missing.csv'
df = pd.read_csv(csv_file_path)

# Step 1: Check for missing data
print("Missing data before handling:")
print(df.isnull().sum())

# Identify numeric columns only
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Option 1: Fill missing values with the mean for numeric columns only
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 2: Standardization of descriptors (excluding non-numeric columns like PDB_ID or SMILES)
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Step 3: Check if any duplicates exist and remove them
df.drop_duplicates(inplace=True)

# Step 4: Display the processed data
print("Processed DataFrame:")
print(df.head())

# Step 5: Save the cleaned DataFrame
df.to_csv('cleaned_descriptors.csv', index=False)
