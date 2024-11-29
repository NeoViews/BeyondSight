"""
dashboard.py
Author: Beyondsight Team
Date: Nov 19, 2024

Description:
This script defines an interactive dashboard using Dash to visualize the results 
of an LSTM model trained on soccer tracking data. It includes functionalities for 
plotting predicted vs actual values and the corresponding errors for selected variables.

Key Features:
- Preprocessing of tracking data.
- LSTM model integration for prediction.
- Dash-based interactive visualization of model results.

Usage:
Run this script to launch the dashboard. Ensure all dependencies are installed, 
and the required dataset (`tracking_data_full_subset2.csv`.etc) and pre-trained model (`best_model.pth`) are available.
"""

# Import required libraries
import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

# Set device to GPU if available, otherwise fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Part 1: Load dataset and preprocessing
# Load and preprocess the dataset
df = pd.read_csv("data/tracking_data_full_subset2.csv")

# Rename unnamed columns to map corresponding X and Y coordinates
# This is needed to clean up the dataset and make column names intuitive
new_columns = []
for i, col in enumerate(df.columns):
    if "Unnamed" in col:
        corresponding_x = df.columns[i - 1]
        new_columns.append(f"{corresponding_x}_Y")  # Rename Y-coordinate columns
    else:
        new_columns.append(col)

# Apply the new column names to the DataFrame
df.columns = new_columns

# Handle NaN values
df = df.fillna(0)  # Fill NaN values with 0, or you can use interpolation


# Function to preprocess data into sequences for LSTM
def preprocess_data(df, sequence_length=50):
    """
    Prepares the dataset for LSTM model training by creating sequences.

    Parameters:
    ----------
    df : pandas.DataFrame
        The dataset containing the input features.
    sequence_length : int, optional
        The length of each sequence (default is 50).

    Returns:
    -------
    np.array
        A NumPy array of sequences, each of length `sequence_length`.
    """
    sequences = []
    for i in range(len(df) - sequence_length):
        sequence = df.iloc[i : i + sequence_length].values  # Extract a sequence of data
        sequences.append(sequence)
    return np.array(sequences)


# Part 2: LSTM predictions
# Define the LSTM model
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

# Initialize and load pre-trained LSTM model
input_size = 62  # Each row has 62 features (31 X-Y coordinate pairs)
hidden_size = 128  # Number of LSTM units
output_size = 62  # Output size should match the input size
num_layers = 2 # Number of stacked LSTM layers in the model for better learning capacity.
learning_rate = 0.001

model = BeyondSightLSTM(input_size, hidden_size, output_size, num_layers).to(
    DEVICE
)  # Move model to the device
model.load_state_dict(
    torch.load("best_model.pth", weights_only=True)
)  # Load the best model
model.eval()

criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Function to get predictions and calculate errors for visualization
def get_predictions_and_errors(df, model, sequence_length=50):
    """
    Generates predictions and calculates prediction errors using the LSTM model.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input dataset containing the tracking data.
    model : BeyondSightLSTM
        The trained LSTM model for predictions.
    sequence_length : int, optional
        The length of each sequence used for prediction (default is 50).

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing timestamps, actual values, predicted values, 
        and prediction errors for each feature.
    """
    sequences = preprocess_data(df, sequence_length)
    X = sequences[:, :-1, :]  # All but the last time step for input
    y_actual = sequences[:, 1:, :]  # Shifted by one for actual output

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_actual_tensor = torch.tensor(y_actual, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y_pred_tensor = model(X_tensor)

    y_pred = y_pred_tensor.cpu().numpy()  # (num_sequences, sequence_length-1, 62)
    y_actual = y_actual_tensor.cpu().numpy()  # (num_sequences, sequence_length-1, 62)

    # Take only the last timestep of each sequence for comparison
    y_pred_last = y_pred[:, -1, :]  # (num_sequences, 62)
    y_actual_last = y_actual[:, -1, :]  # (num_sequences, 62)

    # Calculate the error per variable per sequence (Mean Absolute Error)
    error = np.abs(y_pred_last - y_actual_last)  # (num_sequences, 62)

    # Create a timestamp range for 0 to 90 minutes (or based on the actual duration)
    num_minutes = len(y_pred_last)  # This should match the number of sequences
    timestamps = pd.Series(range(0, num_minutes))  # Minutes from 0 to num_minutes-1

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            **{f"actual_{i}": y_actual_last[:, i] for i in range(input_size)},
            **{f"predicted_{i}": y_pred_last[:, i] for i in range(input_size)},
            **{f"error_{i}": error[:, i] for i in range(input_size)},
        }
    )

    return results_df


# Get predictions and errors for visualization
results_df = get_predictions_and_errors(df, model)


# Export predicted data
columns_to_export = results_df.loc[:, "predicted_0":"predicted_61"]
columns_to_export.to_csv("data/predicted.csv", index=False)

# Part 3: Dashboard
# Initialize Dash app
app = dash.Dash(__name__)

# Step 9: Create a list of variable names based on the DataFrame columns
variables = [
    f"Variable {i}" for i in range(input_size)
]  # Replace with actual variable names if available

# Define app layout
app.layout = html.Div(
    children=[
        html.H1("LSTM Model Results Dashboard"),
        dcc.Dropdown(
            id="variable-dropdown",
            options=[{"label": var, "value": var} for var in variables],
            value=["Variable 0"],  # Default value as a list for multi-select
            multi=True,  # Allow multiple selections
            placeholder="Select variables to visualize",
        ),
        dcc.Graph(id="predicted_vs_actual"),
        dcc.Graph(id="error_plot"),
    ]
)


# Define callback to update the predicted vs actual graph
@app.callback(
    Output("predicted_vs_actual", "figure"), [Input("variable-dropdown", "value")]
)
def update_predicted_vs_actual(selected_vars):
    """
    Updates the Predicted vs Actual graph based on the selected variables.

    Parameters:
    ----------
    selected_vars : list of str
        The list of variables selected in the dropdown menu.

    Returns:
    -------
    plotly.graph_objects.Figure
        The updated figure showing the Predicted vs Actual values for the selected variables.
    """
    if not selected_vars:
        # If no variable is selected, return an empty figure
        return go.Figure()

    fig = make_subplots(
        rows=len(selected_vars),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Predicted vs Actual for {var}" for var in selected_vars],
    )

    for i, var in enumerate(selected_vars):
        var_index = int(var.split(" ")[-1])  # Extract the index from "Variable X"
        fig.add_trace(
            go.Scatter(
                x=results_df["timestamp"],
                y=results_df[f"actual_{var_index}"],
                mode="lines",
                name=f"Actual {var}",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["timestamp"],
                y=results_df[f"predicted_{var_index}"],
                mode="lines",
                name=f"Predicted {var}",
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        height=300 * len(selected_vars),  # Adjust height based on number of subplots
        width=1200,
        title_text="Predicted vs Actual Values",
        showlegend=False,
    )

    return fig


# Define callback to update the error plot
@app.callback(Output("error_plot", "figure"), [Input("variable-dropdown", "value")])
def update_error_plot(selected_vars):
    """
    Updates the error plot based on the selected variables.

    Parameters:
    ----------
    selected_vars : list of str
        The list of variables selected in the dropdown menu.

    Returns:
    -------
    plotly.graph_objects.Figure
        The updated figure showing prediction errors for the selected variables.
    """
    if not selected_vars:
        # If no variable is selected, return an empty figure
        return go.Figure()

    fig = make_subplots(
        rows=len(selected_vars),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Prediction Error for {var}" for var in selected_vars],
    )

    for i, var in enumerate(selected_vars):
        var_index = int(var.split(" ")[-1])  # Extract the index from "Variable X"
        fig.add_trace(
            go.Scatter(
                x=results_df["timestamp"],
                y=results_df[f"error_{var_index}"],
                mode="lines",
                name=f"Error {var}",
                line=dict(color="orange"),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        height=300 * len(selected_vars),  # Adjust height based on number of subplots
        width=1200,
        title_text="Prediction Error",
        showlegend=False,
    )

    return fig


# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
