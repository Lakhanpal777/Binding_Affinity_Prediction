import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the model architecture
def build_model(input_dim):
    """
    Builds and compiles a regression model.
    
    Parameters:
    - input_dim: Number of input features
    
    Returns:
    - Compiled model
    """
    model = Sequential()
    
    # Input Layer + Hidden Layers
    model.add(Dense(128, input_dim=input_dim, activation='relu'))  # First Dense Layer
    model.add(Dropout(0.2))  # Dropout to prevent overfitting
    
    model.add(Dense(64, activation='relu'))  # Second Dense Layer
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))  # Third Dense Layer
    
    # Output Layer
    model.add(Dense(1, activation='linear'))  # Regression output
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Define input dimension
input_dim = 150  # Adjust based on your input features

# Build the model
model = build_model(input_dim)

# Print the model summary
print("\nModel Summary:")
model.summary()

# Save model structure
with open("model_structure.json", "w") as json_file:
    json_file.write(model.to_json())

# Save model weights (untrained)
model.save_weights("untrained_model.weights.h5")

