"""
lstm_model_comparison.py
Author: BeyondInsight Team
Date: Nov 19, 2024

Description:
This script visualizes soccer pitch tracking data and compares actual vs predicted 
positions using an LSTM model. It includes:
- Data preprocessing for sequence creation.
- Visualization of tracking data on a soccer pitch.
- LSTM-based predictions and comparison with actual data.

"""

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Set device to GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Prepare the data for LSTM
def preprocess_data(df, sequence_length=50):
    """
    Prepares the dataset for LSTM model training or prediction by creating sequences.

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


def plot_pitch_v2(
    figax=None,
    field_dimen=(106.0, 68.0),
    field_color="green",
    linewidth=2,
    markersize=20,
):
    """plot_pitch

    Plots a soccer pitch. All distance units converted to meters.

    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20

    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    if figax is None:  # create new pitch
        fig, ax = plt.subplots(figsize=(12, 8))  # create a figure
    else:  # overlay on a previously generated pitch
        fig, ax = figax  # unpack tuple
    # fig,ax = plt.subplots(figsize=(12,8)) # create a figure
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color == "green":
        # if field_color=='black':
        ax.set_facecolor("mediumseagreen")
        # ax.set_facecolor('black')
        lc = "whitesmoke"  # line color
        pc = "w"  # 'spot' colors
    elif field_color == "white":
        lc = "k"
        pc = "k"
    # ALL DIMENSIONS IN m
    # border_dimen = (3,3) # include a border arround of the field of width 3m
    border_dimen = (0, 0)  # include a border arround of the field of width 3m
    meters_per_yard = 0.9144  # unit conversion from yards to meters
    half_pitch_length = field_dimen[0] / 2.0  # length of half pitch
    half_pitch_width = field_dimen[1] / 2.0  # width of half pitch
    signs = [-1, 1]
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8 * meters_per_yard
    box_width = 20 * meters_per_yard
    box_length = 6 * meters_per_yard
    area_width = 44 * meters_per_yard
    area_length = 18 * meters_per_yard
    penalty_spot = 12 * meters_per_yard
    corner_radius = 1 * meters_per_yard
    D_length = 8 * meters_per_yard
    D_radius = 10 * meters_per_yard
    D_pos = 12 * meters_per_yard
    centre_circle_radius = 10 * meters_per_yard
    # plot half way line # center circle

    ax.plot([0, 0], [-half_pitch_width, half_pitch_width], lc, linewidth=linewidth)
    ax.scatter(0.0, 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize)
    y = np.linspace(-1, 1, 50) * centre_circle_radius
    x = np.sqrt(centre_circle_radius**2 - y**2)
    ax.plot(x, y, lc, linewidth=linewidth)
    ax.plot(-x, y, lc, linewidth=linewidth)
    for s in signs:  # plots each line seperately
        # plot pitch boundary
        ax.plot(
            [-half_pitch_length, half_pitch_length],
            [s * half_pitch_width, s * half_pitch_width],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-half_pitch_width, half_pitch_width],
            lc,
            linewidth=linewidth,
        )
        # goal posts & line
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-goal_line_width / 2.0, goal_line_width / 2.0],
            pc + "s",
            markersize=6 * markersize / 20.0,
            linewidth=linewidth,
        )
        # 6 yard box
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [-box_width / 2.0, -box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [
                s * half_pitch_length - s * box_length,
                s * half_pitch_length - s * box_length,
            ],
            [-box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        # penalty area
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [-area_width / 2.0, -area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [
                s * half_pitch_length - s * area_length,
                s * half_pitch_length - s * area_length,
            ],
            [-area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        # penalty spot
        ax.scatter(
            s * half_pitch_length - s * penalty_spot,
            0.0,
            marker="o",
            facecolor=lc,
            linewidth=0,
            s=markersize,
        )
        # corner flags
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius**2 - y**2)
        ax.plot(
            s * half_pitch_length - s * x,
            -half_pitch_width + y,
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            s * half_pitch_length - s * x, half_pitch_width - y, lc, linewidth=linewidth
        )
        # draw the D
        y = (
            np.linspace(-1, 1, 50) * D_length
        )  # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2 - y**2) + D_pos
        ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=linewidth)

    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0] / 2.0 + border_dimen[0]
    ymax = field_dimen[1] / 2.0 + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)
    return fig, ax

def get_predictions_and_actuals(df, model, sequence_length=50):
    """
    Generates predictions and extracts actual values using the LSTM model.

    Parameters:
    ----------
    df : pandas.DataFrame
        The dataset containing input features for prediction.
    model : BeyondSightLSTM
        The pre-trained LSTM model.
    sequence_length : int, optional
        The length of each input sequence (default is 50).

    Returns:
    -------
    tuple of np.array
        Predicted values and actual values for the dataset.
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

    return y_pred, y_actual

# Visualize actual vs. predicted positions on the soccer pitch
def visualize_actual_vs_predicted_on_pitch(df, y_pred, y_actual, field_dimen=(106.0, 68.0)):
    """
    Visualizes actual and predicted positions of players and the ball on a soccer pitch.

    Parameters:
    ----------
    df : pandas.DataFrame
        The dataset containing input features for visualization.
    y_pred : np.array
        Predicted positions for each time step and feature.
    y_actual : np.array
        Actual positions for each time step and feature.
    field_dimen : tuple of float, optional
        Dimensions of the soccer pitch in meters as (length, width) (default is (106, 68)).

    Returns:
    -------
    None
        Displays a dynamic visualization of the actual and predicted positions.
    """
    # Use the plot_pitch_v2 function to plot the pitch
    fig, ax = plot_pitch_v2(field_dimen=field_dimen)
    
    # Extract the actual and predicted X-Y coordinates for players and ball
    num_players = 30  # 30 players: 15 per team (without ball)
    ball_index = 60  # Last two coordinates are for the ball (index 60 and 61)

    for i in range(len(y_pred)):
        # Clear previous frame
        ax.cla()
        
        # Plot the pitch again
        plot_pitch_v2((fig, ax), field_dimen=field_dimen)
        
        # Get actual and predicted positions for this timestep
        actual_positions = y_actual[i, -1, :].reshape(31, 2)  # Last timestep
        predicted_positions = y_pred[i, -1, :].reshape(31, 2)  # Last timestep
        
        # Plot Team A (first 15 players) - actual in green, predicted in red
        for j in range(15):  # First 15 players
            ax.scatter(actual_positions[j, 0], actual_positions[j, 1], c="green", s=100, label="Team A Actual" if i == 0 and j == 0 else "")
            ax.scatter(predicted_positions[j, 0], predicted_positions[j, 1], c="red", s=100, label="Team A Predicted" if i == 0 and j == 0 else "")
        
        # Plot Team B (next 15 players) - actual in blue, predicted in orange
        for j in range(15, 30):  # Next 15 players
            ax.scatter(actual_positions[j, 0], actual_positions[j, 1], c="blue", s=100, label="Team B Actual" if i == 0 and j == 15 else "")
            ax.scatter(predicted_positions[j, 0], predicted_positions[j, 1], c="orange", s=100, label="Team B Predicted" if i == 0 and j == 15 else "")
        
        # Plot Ball - actual in white, predicted in yellow
        ax.scatter(actual_positions[30, 0], actual_positions[30, 1], c="white", s=200, label="Ball Actual" if i == 0 else "")
        ax.scatter(predicted_positions[30, 0], predicted_positions[30, 1], c="yellow", s=200, label="Ball Predicted" if i == 0 else "")
        
        # Add labels and legend
        if i == 0:
            ax.legend(loc="upper right")
        
        # Pause to create an animation effect
        plt.pause(0.1)

    plt.show()

df = pd.read_csv("data/tracking_data_full_subset2.csv")


# Part 2: LSTM predictions
# Define the LSTM model
class BeyondSightLSTM(nn.Module):
    """
    LSTM model for multi-step prediction of player and ball positions.

    Attributes:
    ----------
    input_size : int
        The number of input features (e.g., 62 for X-Y coordinate pairs).
    hidden_size : int
        Number of units in each LSTM layer.
    output_size : int
        Size of the output, matching the input size.
    num_layers : int, optional
        Number of stacked LSTM layers (default is 1).
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initializes the LSTM model with the specified parameters.

        Parameters:
        ----------
        input_size : int
            Number of input features.
        hidden_size : int
            Number of LSTM units in each layer.
        output_size : int
            Size of the output layer.
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
        Performs a forward pass through the model.

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


# Initialize the model, loss function, and optimizer
input_size = 62  # Each row has 62 features (31 X-Y coordinate pairs)
hidden_size = 128  # Number of LSTM units
output_size = 62  # Output size should match the input size
num_layers = 2
learning_rate = 0.001

model = BeyondSightLSTM(input_size, hidden_size, output_size, num_layers).to(DEVICE)  # Move model to the device
model.load_state_dict(
    torch.load("best_model.pth", weights_only=True)
)  # Load the best model
model.eval()

criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Run predictions
y_pred, y_actual = get_predictions_and_actuals(df, model)

# Visualize the actual vs. predicted positions
visualize_actual_vs_predicted_on_pitch(df, y_pred, y_actual)
