# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler

# Step 1: Load Data
# Replace 'your_file.xlsx' with the path to your Excel file
file_path = r'C:\Users\91985\drug-discovery-analysis\deep_learning\final_input_concatenated.xlsx'
data = pd.read_excel(file_path)

# Step 2: Identify Features and Target
# Columns: 'PDB_ID', 'Binding affinity', 'SMILES', and features from '0' to '149'
target_column = "Binding affinity"
feature_columns = list(data.columns[3:])  # Select all columns from '0' to '149'

# Extract features (X) and target variable (y)
X = data[feature_columns].values  # Features (columns '0' to '149')
y = data[target_column].values    # Target variable (Binding affinity)

# Step 3: Split Data into Training, Validation, and Test Sets
# First split into training and temporary sets (70% training, 30% temporary)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split the temporary set into validation and test sets (15% each)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the shapes of the splits to verify
print(f"Training Set: {X_train.shape}, {y_train.shape}")
print(f"Validation Set: {X_val.shape}, {y_val.shape}")
print(f"Test Set: {X_test.shape}, {y_test.shape}")

# Step 4: Normalize Features
# Standardize the data using StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform all splits
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 5: Normalize Target Variable
# Standardize the target variable (Binding affinity)
target_scaler = StandardScaler()

# Fit the scaler on the training target and transform all splits
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Step 6: Save Splits to CSV Files
# Replace 'path_to_save' with your desired directory path
save_path = r'C:\Users\91985\drug-discovery-analysis\deep_learning'  # Update with your desired save directory

# Save features
pd.DataFrame(X_train_scaled).to_csv(f"{save_path}/X_train.csv", index=False)
pd.DataFrame(X_val_scaled).to_csv(f"{save_path}/X_val.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv(f"{save_path}/X_test.csv", index=False)

# Save targets
pd.DataFrame(y_train_scaled).to_csv(f"{save_path}/y_train.csv", index=False, header=["Binding affinity"])
pd.DataFrame(y_val_scaled).to_csv(f"{save_path}/y_val.csv", index=False, header=["Binding affinity"])
pd.DataFrame(y_test_scaled).to_csv(f"{save_path}/y_test.csv", index=False, header=["Binding affinity"])

# Save the scaler for the target variable for inverse transformation later
joblib.dump(target_scaler, f"{save_path}/target_scaler.pkl")  # Save the scaler for later use

# Print confirmation message
print("Data splitting and saving completed successfully.")
