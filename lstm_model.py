import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import wandb
import os

# Initialize WandB and start a new run
wandb.init(
    project="BeyondSight",
    config={
        "learning_rate": 0.001,
        "architecture": "LSTM",
        "dataset": "tracking_data_full_subset2",
        "epochs": 20,
        "hidden_size": 128,
        "input_size": 62,  # Updated input size
        "output_size": 62,  # Output size should match input size
        "num_layers": 2
    }
)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Load dataset
df = pd.read_csv('data/tracking_data_full_subset2.csv')

# Step 2: Rename the unnamed columns to match their corresponding X-coordinate columns
new_columns = []
for i, col in enumerate(df.columns):
    if 'Unnamed' in col:
        corresponding_x = df.columns[i - 1]
        new_columns.append(f"{corresponding_x}_Y")  # Rename Y-coordinate columns
    else:
        new_columns.append(col)

# Apply the new column names to the DataFrame
df.columns = new_columns

# Step 3: Handle NaN values
df = df.fillna(0)  # Fill NaN values with 0, or you can use interpolation

# Step 4: Prepare the data for LSTM
def preprocess_data(df):
    sequences = []
    sequence_length = 50  # Define the length of the sequence

    for i in range(len(df) - sequence_length):
        sequence = df.iloc[i:i+sequence_length].values  # Extract a sequence of data
        sequences.append(sequence)
    
    sequences = np.array(sequences)
    return sequences

# Step 5: Define the LSTM model
class BeyondSightLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BeyondSightLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to output predictions for each timestep
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply fully connected layer to every timestep (not just the last one)
        out = self.fc(out)
        return out

# Step 6: Initialize the model, loss function, and optimizer
input_size = 62  # Each row has 62 features (X-Y coordinate pairs)
hidden_size = 128  # Number of LSTM units
output_size = input_size  # Output size should match the input size
num_layers = 2
learning_rate = 0.001

model = BeyondSightLSTM(input_size, hidden_size, output_size, num_layers).to(device)  # Move model to the device
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 7: Prepare data loaders
def prepare_dataloaders(sequences, batch_size=32):
    X = sequences[:, :-1, :]  # All but the last time step for input
    y = sequences[:, 1:, :]   # Shifted by one for target output
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# Step 8: Train the model with WandB logging and model saving
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    best_loss = float('inf')  # Initialize with a high value
    best_model_path = "best_model.pth"  # Path to save the best model

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
        
        # Log the loss to WandB
        wandb.log({"epoch": epoch+1, "loss": avg_loss})
        
        # Save the model if it has the best performance so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with loss {best_loss}")

# Step 9: Data simulation and training
sequences = preprocess_data(df)
train_loader = prepare_dataloaders(sequences)
train_model(model, train_loader, criterion, optimizer, epochs=wandb.config.epochs)

# Finish the WandB run
wandb.finish()
