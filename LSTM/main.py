import gc
import os
import sys
from typing import Tuple

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----- CONFIG -----
INPUT_DIM = 2  # Only lon, lat - no need for constant Z
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 2  # Only predict lon, lat
# SEQUENCE_LENGTH = 20 → good starting point (20 years back → predict next). You can tune this: try 10, 30, etc.
SEQUENCE_LENGTH = 20
EPOCHS = 100
FORECAST_STEPS = 10
LEARNING_RATE = 0.001

files = [
    os.path.join("./output_segments", f)
    for f in os.listdir("./output_segments")
    if f.endswith(".gpkg")
]


def load_coastline_data(file: str) -> gpd.GeoDataFrame:
    """Load coastline data from files"""
    return gpd.read_file(file)


def extract_coordinates_from_geometry(geometry):
    """Extract (lon, lat) coordinates from MultiLineString geometry, dropping Z coordinate"""
    coords = []
    for line in geometry.geoms:
        # Only take lon, lat - drop the Z coordinate
        coords.extend([(coord[0], coord[1]) for coord in line.coords])
    return np.array(coords)


def prepare_lstm_sequences_single_segment(
    segment: gpd.GeoDataFrame, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training from a single segment
    Returns: (X, y) where X is input sequences and y is target coordinates
    """
    X, y = [], []

    # Extract coordinates for each timestep
    coords_list = []
    for _, row in segment.iterrows():
        coords = extract_coordinates_from_geometry(row.geometry)
        # Use mean coordinates for each timestep (you could also use other aggregation)
        mean_coords = np.mean(coords, axis=0)  # [lon, lat]
        coords_list.append(mean_coords)

    coords_array = np.array(coords_list)

    # Create sequences
    for i in range(len(coords_array) - sequence_length):
        X.append(coords_array[i : i + sequence_length])
        y.append(coords_array[i + sequence_length])

    return np.array(X), np.array(y)


class CoastlineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(CoastlineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        out = self.fc(out[:, -1, :])  # predict from last timestep
        return out


def train_model_on_segment(model, optimizer, criterion, X, y, epochs=10):
    """Train the model on a single segment's data"""
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return loss.item()


def forecast(model, init_seq, steps):
    """Generate future predictions using the trained model"""
    preds = []
    seq = init_seq.clone()

    for _ in range(steps):
        pred = model(seq.unsqueeze(0))  # shape [1, 2] -> (lon, lat)
        preds.append(pred.detach().numpy())
        # append prediction to sequence, drop oldest timestep
        seq = torch.cat([seq[1:], pred], dim=0)

    return np.array(preds)


# def test():
#     df = load_coastline_data(files[0])

#     print(df.iloc[0].Date)
#     print(df.iloc[0].Segment)
#     print(df.iloc[0].geometry)

#     num_coordinate_points = {}

#     for i in range(len(df)):
#         geom = df.iloc[i].geometry
#         num_points = 0

#         for j, line in enumerate(geom.geoms):
#             num_points += len(list(line.coords))
#             print(f"LineString {j}: {list(line.coords)} coordinate points")

#         if num_points not in num_coordinate_points:
#             num_coordinate_points[num_points] = 1
#         else:
#             num_coordinate_points[num_points] += 1

#     print(f"dict of number of coordinate points: {num_coordinate_points}")

#     import json

#     with open("num_coordinate_points.json", "w") as f:
#         json.dump(num_coordinate_points, f, indent=2)

#     sys.exit()


if __name__ == "__main__":
    # test()
    if files:
        print(f"Found {len(files)} segment files to process")

        # Create model and optimizer
        model = CoastlineLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        # Process each segment sequentially
        for i, file_path in enumerate(files):
            print(
                f"\n--- Processing segment {i+1}/{len(files)}: {os.path.basename(file_path)} ---"
            )

            # Load single segment
            segment = load_coastline_data(file_path)
            print(f"Segment shape: {segment.shape}")

            # Prepare sequences for this segment only
            X, y = prepare_lstm_sequences_single_segment(segment, SEQUENCE_LENGTH)

            if len(X) == 0:
                print(f"No sequences generated for segment {i+1}, skipping...")
                continue

            print(f"Generated {len(X)} sequences from segment {i+1}")

            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            # Train model on this segment
            print(f"Training on segment {i+1}...")
            final_loss = train_model_on_segment(
                model, optimizer, criterion, X_tensor, y_tensor, epochs=EPOCHS
            )
            print(f"Final loss for segment {i+1}: {final_loss:.6f}")

            # Clear memory
            del segment, X, y, X_tensor, y_tensor
            gc.collect()

            # Test forecasting on first segment only
            if i == 0:
                print("\nTesting forecasting...")
                # Reload first segment for forecasting test
                test_segment = load_coastline_data(files[0])
                X_test, _ = prepare_lstm_sequences_single_segment(
                    test_segment, sequence_length=SEQUENCE_LENGTH
                )
                if len(X_test) > 0:
                    forecast_steps = FORECAST_STEPS
                    predictions = forecast(
                        model, torch.FloatTensor(X_test[0]), forecast_steps
                    )
                    print(f"Forecast shape: {predictions.shape}")
                    print(f"Sample prediction: {predictions[0]}")
                del test_segment, X_test
                gc.collect()

        print("\nTraining completed on all segments!")
    else:
        print("No .gpkg files found in output_segments directory")
