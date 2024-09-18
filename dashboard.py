import torch
import torch.nn as nn 
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set device to GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
def preprocess_data(df, sequence_length=50):
    sequences = []
    for i in range(len(df) - sequence_length):
        sequence = df.iloc[i:i+sequence_length].values  # Extract a sequence of data
        sequences.append(sequence)
    return np.array(sequences)

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
input_size = 62  # Each row has 62 features (31 X-Y coordinate pairs)
hidden_size = 128  # Number of LSTM units
output_size = 62  # Output size should match the input size
num_layers = 2
learning_rate = 0.001

model = BeyondSightLSTM(input_size, hidden_size, output_size, num_layers).to(DEVICE)  # Move model to the device
model.load_state_dict(torch.load("best_model.pth", weights_only=True))  # Load the best model
model.eval()

criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Step 7: Make predictions using the loaded model
def get_predictions_and_errors(df, model, sequence_length=50):
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
    
    # Create a DataFrame to store the results
    timestamps = pd.date_range(start="2023-01-01", periods=len(y_pred_last), freq="min")  # Example timestamps
    results_df = pd.DataFrame({
        "timestamp": timestamps,
        **{f"actual_{i}": y_actual_last[:,i] for i in range(input_size)},
        **{f"predicted_{i}": y_pred_last[:,i] for i in range(input_size)},
        **{f"error_{i}": error[:,i] for i in range(input_size)}
    })
    
    return results_df

# Get predictions and errors for visualization
results_df = get_predictions_and_errors(df, model)

# Step 8: Initialize Dash app
app = dash.Dash(__name__)

# Step 9: Create a list of variable names based on the DataFrame columns
variables = [f"Variable {i}" for i in range(input_size)]  # Replace with actual variable names if available

# Define app layout
app.layout = html.Div(
    children=[
        html.H1("LSTM Model Results Dashboard"),
        dcc.Dropdown(
            id="variable-dropdown",
            options=[{"label": var, "value": var} for var in variables],
            value=["Variable 0"],  # Default value as a list for multi-select
            multi=True,  # Allow multiple selections
            placeholder="Select variables to visualize"
        ),
        dcc.Graph(id="predicted_vs_actual"),
        dcc.Graph(id="error_plot"),
    ]
)

# Define callback to update the predicted vs actual graph
@app.callback(
    Output("predicted_vs_actual", "figure"),
    [Input("variable-dropdown", "value")]
)
def update_predicted_vs_actual(selected_vars):
    if not selected_vars:
        # If no variable is selected, return an empty figure
        return go.Figure()
    
    fig = make_subplots(
        rows=len(selected_vars),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Predicted vs Actual for {var}" for var in selected_vars]
    )
    
    for i, var in enumerate(selected_vars):
        var_index = int(var.split(" ")[-1])  # Extract the index from "Variable X"
        fig.add_trace(
            go.Scatter(
                x=results_df["timestamp"],
                y=results_df[f"actual_{var_index}"],
                mode="lines",
                name=f"Actual {var}"
            ),
            row=i+1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=results_df["timestamp"],
                y=results_df[f"predicted_{var_index}"],
                mode="lines",
                name=f"Predicted {var}"
            ),
            row=i+1,
            col=1
        )
    
    fig.update_layout(
        height=300 * len(selected_vars),  # Adjust height based on number of subplots
        width=1200,
        title_text="Predicted vs Actual Values",
        showlegend=False
    )
    
    return fig

# Define callback to update the error plot
@app.callback(
    Output("error_plot", "figure"),
    [Input("variable-dropdown", "value")]
)
def update_error_plot(selected_vars):
    if not selected_vars:
        # If no variable is selected, return an empty figure
        return go.Figure()
    
    fig = make_subplots(
        rows=len(selected_vars),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Prediction Error for {var}" for var in selected_vars]
    )
    
    for i, var in enumerate(selected_vars):
        var_index = int(var.split(" ")[-1])  # Extract the index from "Variable X"
        fig.add_trace(
            go.Scatter(
                x=results_df["timestamp"],
                y=results_df[f"error_{var_index}"],
                mode="lines",
                name=f"Error {var}",
                line=dict(color='orange')
            ),
            row=i+1,
            col=1
        )
    
    fig.update_layout(
        height=300 * len(selected_vars),  # Adjust height based on number of subplots
        width=1200,
        title_text="Prediction Error",
        showlegend=False
    )
    
    return fig

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
