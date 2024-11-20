"""
lstm_model.py
Author: BeyondInsight Team
Date: Nov 19,2024

Description:
This script implements the training pipeline for an LSTM model to predict soccer 
tracking data. It integrates with Weights and Biases (WandB) for experiment tracking 
and logs training metrics for analysis.

Key Features:
- Data preprocessing for LSTM training.
- Definition of an LSTM model with configurable parameters.
- Model training loop with WandB integration for logging and monitoring.
- Automatic saving of the best-performing model.

Usage:
Run this script to train the LSTM model. Ensure the tracking data 
(`tracking_data_full_subset2.csv`) is available in the `data` directory.
"""

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
    """
    Prepares the dataset for LSTM model training by creating sequences.

    Parameters:
    ----------
    df : pandas.DataFrame
        The dataset containing the input features.

    Returns:
    -------
    np.array
        A NumPy array of sequences, each of a fixed sequence length.
    """
    sequences = []
    sequence_length = 50  # Define the length of the sequence

    for i in range(len(df) - sequence_length):
        sequence = df.iloc[i:i+sequence_length].values  # Extract a sequence of data
        sequences.append(sequence)
    
    sequences = np.array(sequences)
    return sequences

# Step 5: Define the LSTM model
class BeyondSightLSTM(nn.Module):
    """
    LSTM model for multi-step prediction of player and ball positions.

    Attributes:
    ----------
    input_size : int
        The number of features in the input (e.g., 62 for X-Y coordinate pairs).
    hidden_size : int
        The number of units in each LSTM layer.
    output_size : int
        The size of the output (should match the input size).
    num_layers : int, optional
        The number of stacked LSTM layers (default is 1).
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initializes the LSTM model with the specified parameters.

        Parameters:
        ----------
        input_size : int
            Number of features in the input data.
        hidden_size : int
            Number of units in the LSTM layer.
        output_size : int
            Size of the model's output.
        num_layers : int, optional
            Number of stacked LSTM layers (default is 1).
        """
        super(BeyondSightLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to output predictions for each timestep
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass for the LSTM model.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, output_size).
        """
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
    """
    Prepares data loaders for training the LSTM model.

    Parameters:
    ----------
    sequences : np.array
        Array of sequences created from the dataset.
    batch_size : int, optional
        Number of samples per batch (default is 32).

    Returns:
    -------
    torch.utils.data.DataLoader
        DataLoader for batching and shuffling the data.
    """
    X = sequences[:, :-1, :]  # All but the last time step for input
    #print(sequences[:, :-1, :])
    y = sequences[:, 1:, :]   # Shifted by one for target output
    # print(sequences[:, 1:, :])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# Step 8: Train the model with WandB logging and model saving
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """
    Trains the LSTM model with logging and model saving.

    Parameters:
    ----------
    model : BeyondSightLSTM
        The LSTM model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader containing training data.
    criterion : torch.nn.Module
        Loss function for the training process.
    optimizer : torch.optim.Optimizer
        Optimizer for updating the model parameters.
    epochs : int, optional
        Number of epochs for training (default is 10).

    Returns:
    -------
    None
    """
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
