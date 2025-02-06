import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 1: Load Data
data_path = r'C:\Users\91985\drug-discovery-analysis\deep_learning'
X_train = pd.read_csv(f"{data_path}/X_train.csv").values
X_val = pd.read_csv(f"{data_path}/X_val.csv").values
X_test = pd.read_csv(f"{data_path}/X_test.csv").values
y_train = pd.read_csv(f"{data_path}/y_train.csv").values.ravel()
y_val = pd.read_csv(f"{data_path}/y_val.csv").values.ravel()
y_test = pd.read_csv(f"{data_path}/y_test.csv").values.ravel()

# Step 2: Define Model Architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Step 3: Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 4: Define Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    "best_model.weights.h5", 
    save_best_only=True, 
    save_weights_only=True,  # Save only weights in .h5 format
    monitor='val_loss'
)

# Step 5: Train the Model
batch_size = 32
epochs = 100
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Step 6: Evaluate the Model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Step 7: Save Final Model
model.save("final_model.keras")
print("Final model saved as 'final_model.keras'")
