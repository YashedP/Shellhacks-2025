import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd
import os

# === Data Preparation ===

files = [
    os.path.join("../data/historical", f)
    for f in os.listdir("../data/historical")
    if f.endswith(".gpkg")
]

def prepare_transformer_sequences_with_deltas(segment, sequence_length=20):
    coords_list = []
    for _, row in segment.iterrows():
        coords = extract_coordinates_from_geometry(row.geometry)
        mean_coords = np.mean(coords, axis=0)  # [lon, lat]
        coords_list.append(mean_coords)
    coords_array = np.array(coords_list)

    # Normalize to [0, 1]
    min_vals = coords_array.min(axis=0)
    max_vals = coords_array.max(axis=0)
    range_vals = max_vals - min_vals
    norm_coords = (coords_array - min_vals) / range_vals

    X, y = [], []
    for i in range(len(norm_coords) - sequence_length - 1):
        seq = norm_coords[i : i + sequence_length]
        last_point = norm_coords[i + sequence_length - 1]
        next_point = norm_coords[i + sequence_length]

        delta = next_point - last_point  # Δlon, Δlat
        
        X.append(seq)
        y.append(delta)

    return np.array(X), np.array(y), min_vals, range_vals

# === Transformer Model ===

class CoastlineTransformerRegression(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1):
        super(CoastlineTransformerRegression, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(d_model, input_dim)  # Output delta vector

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, d_model]
        x = self.transformer_encoder(x)  # [seq_len, batch_size, d_model]
        x = x[-1, :, :]  # Take last timestep's output
        out = self.regressor(x)  # [batch_size, input_dim]
        return out

# === Training ===

def train_model(model, optimizer, criterion, X_train, y_train, epochs=50, batch_size=64, range_vals=None):
    model.train()
    dataset_size = len(X_train)
    for epoch in range(epochs):
        permutation = np.random.permutation(dataset_size)
        epoch_loss = 0
        epoch_mse_meters = 0
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = torch.FloatTensor(X_train[indices])
            batch_y = torch.FloatTensor(y_train[indices])

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)

            # Calculate RMSE in meters for this batch
            # Denormalize: delta in degrees
            denorm_preds = outputs.detach().cpu().numpy() * range_vals
            denorm_targets = batch_y.cpu().numpy() * range_vals
            mse_deg = np.mean((denorm_preds - denorm_targets) ** 2)
            rmse_meters = np.sqrt(mse_deg) * 111000  # degrees to meters approx
            epoch_mse_meters += rmse_meters * len(indices)

        avg_loss = epoch_loss / dataset_size
        avg_rmse_meters = epoch_mse_meters / dataset_size

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, RMSE: {avg_rmse_meters:.2f} meters")


# === Forecasting ===

def forecast(model, init_seq, forecast_steps, min_vals, range_vals):
    """
    Predict future average points by iteratively adding predicted delta vectors.
    init_seq: normalized sequence of points [seq_len, 2]
    """
    model.eval()
    seq = init_seq.clone()  # [seq_len, 2]
    preds = []

    with torch.no_grad():
        for _ in range(forecast_steps):
            input_seq = seq.unsqueeze(0)  # Add batch dim: [1, seq_len, 2]
            delta_pred = model(input_seq).squeeze(0)  # [2]
            new_point = seq[-1] + delta_pred
            preds.append(new_point.numpy())
            seq = torch.cat([seq[1:], new_point.unsqueeze(0)], dim=0)  # Slide window

    preds = np.array(preds)
    # Denormalize
    preds_denorm = preds * range_vals + min_vals
    return preds_denorm

# === Example usage ===
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


if __name__ == "__main__":
    # Assume `segment` is a GeoDataFrame loaded elsewhere (your coastline segment)

    SEQUENCE_LENGTH = 20
    EPOCHS = 50
    LEARNING_RATE = 0.005
    for i, file_path in enumerate(files):
        print(
            f"\n--- Processing segment {i+1}/{len(files)}: {os.path.basename(file_path)} ---"
        )

        # Load single segment
        segment = load_coastline_data(file_path)    # Prepare sequences and deltas
        X, y, min_vals, range_vals = prepare_transformer_sequences_with_deltas(segment, SEQUENCE_LENGTH)

        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")

        print(f"Prepared {len(X)} sequences")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Initialize model, loss, optimizer
        model = CoastlineTransformerRegression()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train
        train_model(model, optimizer, criterion, X, y, epochs=EPOCHS, batch_size=64, range_vals=range_vals)

        # Forecast example
        init_seq = torch.FloatTensor(X[0])  # Use first sequence as seed
        forecast_steps = 10
        predictions = forecast(model, init_seq, forecast_steps, min_vals, range_vals)

        print(f"Forecasted {forecast_steps} future average points:")
        print(predictions)
