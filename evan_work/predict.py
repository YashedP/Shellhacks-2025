# predict.py
import os, json, argparse
import numpy as np
import torch
from shapely.geometry import LineString
import geopandas as gpd
from pyproj import CRS, Transformer

# ---- helpers ----
def spatial_resample_to_N(arr, N_new):
    """
    Resample along-shore points from N_old -> N_new for an entire series.
    Uses arclength-parametrized linear interpolation per timestep.
    arr: [T, N_old, 2] (meters). Returns [T, N_new, 2].
    """
    T, N_old, _ = arr.shape
    if N_old == N_new:
        return arr.astype(np.float32)

    out = np.empty((T, N_new, 2), dtype=np.float32)
    for t in range(T):
        pts = arr[t]                             # [N_old,2]
        diffs = pts[1:] - pts[:-1]
        seglen = np.sqrt((diffs**2).sum(axis=1))
        s = np.concatenate(([0.0], np.cumsum(seglen)))  # [N_old]
        L = float(s[-1])
        if not np.isfinite(L) or L == 0.0:
            out[t] = np.repeat(pts[:1], N_new, axis=0)
            continue
        s_new = np.linspace(0.0, L, N_new)
        out[t, :, 0] = np.interp(s_new, s, pts[:, 0])
        out[t, :, 1] = np.interp(s_new, s, pts[:, 1])
    return out

def utm_to_wgs84_line(xy_m, utm_authority):
    # xy_m: [N,2] meters; utm_authority like ["EPSG","32617"]
    utm = CRS.from_authority(*utm_authority)
    wgs84 = CRS.from_epsg(4326)
    inv = Transformer.from_crs(utm, wgs84, always_xy=True)
    lon, lat = inv.transform(xy_m[:,0], xy_m[:,1])
    return LineString(np.stack([lon, lat], axis=1))

# ---- model (matches shorelinedata.py) ----
import torch.nn as nn
class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        import math, torch
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)

class CoastlineTransformer(nn.Module):
    def __init__(self, n_points, L_out=1, d_model=256, nhead=8, num_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        in_dim = n_points * 2
        self.in_dim = in_dim
        self.L_out = L_out
        self.input_proj = nn.Linear(in_dim, d_model)
        self.posenc = TimePositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, L_out * in_dim)
        )
    def forward(self, x):
        h = self.input_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        pooled = h[:, -1, :]
        out = self.head(pooled)
        return out.view(x.size(0), self.L_out, self.in_dim)

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Predict next K shoreline steps from a trained model.")
    p.add_argument("--npy",  required=True, help="preproc\\segment_XX_coords.npy")
    p.add_argument("--meta", help="preproc\\segment_XX_meta.json (for WGS84 output)")
    p.add_argument("--ckpt", default=os.path.join("models", "coast_tr_best.pt"))
    p.add_argument("--out_dir", default="output_segments")
    p.add_argument("--steps", type=int, default=1, help="K-step rollout (requires L_out==1)")
    p.add_argument("--abs_target", action="store_true",
                   help="set if model was trained to predict absolutes instead of deltas")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load checkpoint (PyTorch 2.6: weights_only=False for your own file)
    try:
        ck = torch.load(args.ckpt, map_location="cpu")  # try default first
    except Exception:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    stats = ck["stats"]
    cfg   = ck["cfg"]
    N     = int(cfg["N"])
    L_in  = int(cfg["L_in"])
    L_out = int(cfg["L_out"])

    mean = np.array(stats["mean"], dtype=np.float32)  # shape (2,)
    std  = np.array(stats["std"],  dtype=np.float32)

    arr = np.load(args.npy)  # [T, N_data, 2] in meters (UTM)
    if arr.shape[1] != N:
        print(f"[INFO] Resampling along-shore points: data N={arr.shape[1]} -> model N={N}")
        arr = spatial_resample_to_N(arr, N)

    if arr.shape[0] < L_in:
        raise ValueError(f"Not enough timesteps: have T={arr.shape[0]} but need L_in={L_in}")

    # Build model & load weights
    model = CoastlineTransformer(n_points=N, L_out=L_out,
                                 d_model=cfg["d_model"], nhead=cfg["nhead"],
                                 num_layers=cfg["num_layers"], d_ff=cfg["d_ff"])
    model.load_state_dict(ck["model"])
    model.eval()

    # Context window
    ctx = arr[-L_in:].copy()  # [L_in, N, 2]
    preds = []

    with torch.no_grad():
        if args.steps == 1:
            x_norm = (ctx - mean) / std
            xb = torch.tensor(x_norm, dtype=torch.float32).view(1, L_in, -1)
            out = model(xb)[0, 0].view(N, 2).cpu().numpy()
            next_xy = (out * std + mean) if args.abs_target else (ctx[-1] + out * std)
            preds = np.expand_dims(next_xy, axis=0)  # [1, N, 2]
        else:
            if L_out != 1:
                print(f"[WARN] Model L_out={L_out}; rollout assumes one-step predictions. Continuing anyway.")
            for _ in range(args.steps):
                x_norm = (ctx - mean) / std
                xb = torch.tensor(x_norm, dtype=torch.float32).view(1, L_in, -1)
                out = model(xb)[0, 0].view(N, 2).cpu().numpy()
                next_xy = (out * std + mean) if args.abs_target else (ctx[-1] + out * std)
                preds.append(next_xy)
                ctx = np.concatenate([ctx[1:], next_xy[None]], axis=0)
            preds = np.stack(preds, axis=0)  # [K, N, 2]

    # Save outputs
    base = os.path.splitext(os.path.basename(args.npy))[0].replace("_coords", "")
    if args.steps == 1:
        np.save(os.path.join(args.out_dir, f"{base}_pred_xy.npy"), preds[0])
    else:
        np.save(os.path.join(args.out_dir, f"{base}_rollout_xy.npy"), preds)

    # Optional: write GeoPackage in WGS84
    if args.meta and os.path.exists(args.meta):
        meta = json.load(open(args.meta, "r"))
        if args.steps == 1:
            line_ll = utm_to_wgs84_line(preds[0], meta["utm"])
            gdf = gpd.GeoDataFrame({"segment":[meta.get("segment", base)],
                                    "predicted":[True]},
                                   geometry=[line_ll], crs="EPSG:4326")
            out_gpkg = os.path.join(args.out_dir, f"{base}_pred.gpkg")
            gdf.to_file(out_gpkg, driver="GPKG")
            print("Wrote", out_gpkg)
        else:
            geoms = [utm_to_wgs84_line(preds[i], meta["utm"]) for i in range(preds.shape[0])]
            gdf = gpd.GeoDataFrame({"segment":[meta.get("segment", base)]*preds.shape[0],
                                    "step": list(range(1, preds.shape[0]+1))},
                                   geometry=geoms, crs="EPSG:4326")
            out_gpkg = os.path.join(args.out_dir, f"{base}_rollout.gpkg")
            gdf.to_file(out_gpkg, driver="GPKG")
            print("Wrote", out_gpkg)

    print("Done.")

if __name__ == "__main__":
    main()
