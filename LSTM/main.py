import gc
import os
import sys
from typing import Tuple
from shapely.geometry import LineString, MultiLineString

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----- CONFIG -----
INPUT_DIM = 2  # Only lon, lat - no need for constant Z
HIDDEN_DIM = 128
NUM_LAYERS = 4
OUTPUT_DIM = 2  # Only predict lon, lat
# SEQUENCE_LENGTH = 20 → good starting point (20 years back → predict next). You can tune this: try 10, 30, etc.
SEQUENCE_LENGTH = 20
EPOCHS = 100
FORECAST_STEPS = 10
LEARNING_RATE = 0.001

files = [
    os.path.join("../data/historical", f)
    for f in os.listdir("../data/historical")
    if f.endswith(".gpkg")
]

def haversine_distance(coord1, coord2):
    """
    Calculate Haversine distance between two points in (lon, lat)
    Returns distance in meters
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)

    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def load_coastline_data(file: str) -> gpd.GeoDataFrame:
    """Load coastline data from files"""
    return gpd.read_file(file)


def extract_coordinates_from_geometry(geometry):
    """Extract (lon, lat) coordinates from LineString or MultiLineString geometry, dropping Z coordinate"""
    coords = []

    if isinstance(geometry, MultiLineString):
        for line in geometry.geoms:
            coords.extend([(coord[0], coord[1]) for coord in line.coords])
    elif isinstance(geometry, LineString):
        coords.extend([(coord[0], coord[1]) for coord in geometry.coords])
    else:
        print(f"Unsupported geometry type: {type(geometry)}")
    
    return np.array(coords)


def prepare_lstm_sequences_single_segment(
    segment: gpd.GeoDataFrame, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = [], []
    coords_list = []

    for _, row in segment.iterrows():
        coords = extract_coordinates_from_geometry(row.geometry)
        mean_coords = np.mean(coords, axis=0)  # [lon, lat]
        coords_list.append(mean_coords)

    coords_array = np.array(coords_list)

    # --- Normalize ---
    min_vals = coords_array.min(axis=0)  # [min_lon, min_lat]
    max_vals = coords_array.max(axis=0)
    range_vals = max_vals - min_vals
    coords_array = (coords_array - min_vals) / range_vals  # Normalize

    for i in range(len(coords_array) - sequence_length):
        X.append(coords_array[i : i + sequence_length])
        y.append(coords_array[i + sequence_length])

    return np.array(X), np.array(y), min_vals, range_vals


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


def forecast(model, init_seq, steps, min_vals, range_vals):
    preds = []
    seq = init_seq.clone()

    for _ in range(steps):
        pred = model(seq.unsqueeze(0))  # shape [1, 2]
        preds.append(pred.detach().numpy())
        seq = torch.cat([seq[1:], pred], dim=0)

    preds = np.array(preds).squeeze(1)  # shape [steps, 2]

    # --- Denormalize ---
    preds_denorm = preds * range_vals + min_vals
    return preds_denorm



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
            X, y, min_vals, range_vals = prepare_lstm_sequences_single_segment(segment, SEQUENCE_LENGTH)

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
            print(f"Final loss (normalized MSE) for segment {i+1}: {final_loss:.6f}")

            # --- Print RMSE in degrees and meters ---
            rmse_deg = np.sqrt(final_loss)
            rmse_meters = rmse_deg * 111000  # approx conversion (1° ≈ 111 km)
            print(f"RMSE: {rmse_deg:.6f} degrees ≈ {rmse_meters:.2f} meters")

            # Test forecasting on first segment only
            if i == 0:
                print("\nTesting forecasting...")
                test_segment = load_coastline_data(files[0])
                X_test, y_test, min_vals_test, range_vals_test = prepare_lstm_sequences_single_segment(
                    test_segment, sequence_length=SEQUENCE_LENGTH
                )

                if len(X_test) > 0:
                    forecast_steps = FORECAST_STEPS
                    predictions = forecast(
                        model,
                        torch.FloatTensor(X_test[0]),
                        forecast_steps,
                        min_vals_test,
                        range_vals_test,
                    )
                    print(f"Forecast shape: {predictions.shape}")
                    print(f"Sample prediction: {predictions[0]}")

                    # --- Print prediction error in meters ---
                    true_coord = (X_test[0][-1] * range_vals_test + min_vals_test)
                    pred_coord = predictions[0]
                    error_m = haversine_distance(true_coord, pred_coord)
                    print(f"Distance error for 1st forecast step: {error_m:.2f} meters")

                del test_segment, X_test
                gc.collect()


        print("\nTraining completed on all segments!")
    else:
        print("No .gpkg files found in output_segments directory")
