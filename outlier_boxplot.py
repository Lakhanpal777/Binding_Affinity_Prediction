import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix for the OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# File path for the concatenated output
original_file_path = r"C:\Users\91985\drug-discovery-analysis\deep_learning\final_concatenated_output.pt"

try:
    # Load the PyTorch tensor file
    data = torch.load(original_file_path)

    # Detach the tensor and convert to NumPy
    data_numpy = data.detach().numpy()

    # Convert to a Pandas DataFrame
    # Assuming the tensor is 2D with rows as samples and columns as features
    df = pd.DataFrame(data_numpy)

    # Generate a box plot for each column
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, orient='h')
    plt.title('Box Plot of Concatenated Output')
    plt.xlabel('Values')
    plt.ylabel('Features')
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
