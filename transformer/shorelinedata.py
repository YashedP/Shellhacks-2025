import os, glob, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import LineString
import geopandas as gpd
from pyproj import CRS, Transformer
from tqdm.auto import tqdm
import time, csv

# -----------------------------
# Dataset: windows + normalization + delta targets
# -----------------------------
import re
import pandas as pd

def parse_cadence_to_seconds(cadence):
    if cadence is None or str(cadence).strip() == "": return None
    if isinstance(cadence, (int, float)): return int(cadence)
    s = str(cadence).strip().upper()
    m = re.fullmatch(r"(\d+)\s*([SMHDW])", s)
    if m:
        n, u = int(m.group(1)), m.group(2)
        mult = dict(S=1, M=60, H=3600, D=86400, W=604800)[u]
        return n * mult
    return int(pd.to_timedelta(s).total_seconds())


def resample_to_cadence(npy_path, dt_seconds, cache_dir="cache"):
    """
    Resample preproc/<name>_coords.npy to a uniform cadence using the matching
    <name>_meta.json 'dates'. Saves to cache/<name>_coords_dt{dt}s.npy (+ meta).
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(npy_path)
    out_npy  = os.path.join(cache_dir, base.replace("_coords.npy", f"_coords_dt{dt_seconds}s.npy"))
    out_meta = out_npy.replace("_coords_dt", "_meta_dt").replace(".npy", ".json")
    if os.path.exists(out_npy):
        return out_npy

    meta_path = npy_path.replace("_coords.npy", "_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta for {npy_path}")

    meta  = json.load(open(meta_path, "r"))
    dates = pd.to_datetime(meta["dates"])
    arr   = np.load(npy_path)  # [T,N,2]

    # --- sort by time ---
    vals  = dates.values
    order = np.argsort(vals)
    dates = pd.DatetimeIndex(vals[order])
    arr   = arr[order]

    # --- drop duplicate timestamps (keep first) ---
    # (np.unique keeps first occurrence by default)
    uniq_vals, uniq_idx = np.unique(dates.values, return_index=True)
    if len(uniq_idx) != len(dates):
        dates = pd.DatetimeIndex(uniq_vals)
        arr   = arr[uniq_idx]

    if len(dates) < 2:
        # too short to resample — just mirror input
        np.save(out_npy, arr.astype(np.float32))
        json.dump(meta, open(out_meta, "w"))
        return out_npy

    # --- build uniform grid in seconds ---
    t   = (dates.view("int64") // 10**9).astype(np.int64)  # seconds since epoch
    t0, t1 = int(t[0]), int(t[-1])
    new_t   = np.arange(t0, t1 + dt_seconds, dt_seconds, dtype=np.int64)

    # --- linear interpolation per point/coord ---
    T, N, _ = arr.shape
    res = np.empty((len(new_t), N, 2), dtype=np.float32)
    for j in range(N):
        res[:, j, 0] = np.interp(new_t, t, arr[:, j, 0])
        res[:, j, 1] = np.interp(new_t, t, arr[:, j, 1])

    np.save(out_npy, res)
    meta2 = dict(meta)
    meta2["dates"] = pd.to_datetime(new_t, unit="s").astype(str).tolist()
    json.dump(meta2, open(out_meta, "w"))
    return out_npy


class ShorelineDataset(Dataset):
    def __init__(self, npy_paths, L_in=8, L_out=1, stride=1, stats=None,
                 gap=1, cadence_seconds=None, cache_dir="cache"):
        self.L_in, self.L_out, self.gap = L_in, L_out, gap
        self.samples, self.paths = [], []
        # resolve (optionally resampled) paths
        resolved = []
        for p in npy_paths:
            rp = resample_to_cadence(p, cadence_seconds, cache_dir) if cadence_seconds else p
            resolved.append(rp)
            T = np.load(rp, mmap_mode="r").shape[0]
            max_s = T - (L_in + gap + L_out) + 1
            if max_s > 0:
                self.paths.append(rp)
                for s in range(0, max_s, stride):
                    self.samples.append((rp, s))

        # normalization
        if stats is None:
            ssum = np.zeros(2, np.float64); ssum2 = np.zeros(2, np.float64); count = 0
            for rp in self.paths:
                flat = np.load(rp, mmap_mode="r").reshape(-1, 2)
                ssum += flat.sum(0); ssum2 += (flat**2).sum(0); count += flat.shape[0]
            mean = ssum / count
            var  = ssum2 / count - mean**2
            std  = np.sqrt(np.clip(var, 1e-8, None))
            self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        else:
            self.mean, self.std = stats["mean"], stats["std"]

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, s = self.samples[i]
        arr = np.load(path, mmap_mode="r")                 # [T,N,2]
        x = arr[s:s+self.L_in]                             # [L_in,N,2]
        t0 = s + self.L_in - 1 + self.gap                  # predict t+gap
        y = arr[t0:t0+self.L_out]                          # [L_out,N,2]
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        last = x[-1]; y = y - last
        x = torch.tensor(x, dtype=torch.float32).view(self.L_in, -1)
        y = torch.tensor(y, dtype=torch.float32).view(self.L_out, -1)
        return x, y


# -----------------------------
# Model
# -----------------------------
class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # [B,L,D]
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

    def forward(self, x):  # x: [B,L_in,in_dim]
        h = self.input_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        pooled = h[:, -1, :]
        out = self.head(pooled)
        return out.view(x.size(0), self.L_out, self.in_dim)

# -----------------------------
# Training / Evaluation
# -----------------------------
def train_one_epoch(model, loader, opt, device, epoch=0, log_interval=10, use_pbar=True,
                    stats=None, n_points=None, loss_scale=1.0, log_rmse_m=False,
                    optimize_meter_mse=False):
    model.train()
    total_raw = 0.0
    n = len(loader.dataset)
    start = time.time()
    iterator = enumerate(loader)
    if use_pbar:
        iterator = tqdm(iterator, total=len(loader), desc=f"train {epoch:02d}", leave=False)

    # prepare std for meter-space logging/loss
    std_rep = None
    if log_rmse_m or optimize_meter_mse:
        std = torch.tensor(stats["std"], device=device, dtype=torch.float32)  # [2]
        std_rep = std.repeat(n_points).view(1, 1, -1)                         # [1,1,N*2]

    roll_scaled = None
    for i, (xb, yb) in iterator:
        t_batch = time.time()
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        err = pred - yb

        # training objective
        if optimize_meter_mse and std_rep is not None:
            loss = ((err * std_rep) ** 2).mean()     # meters^2
        else:
            loss = (err ** 2).mean()                 # normalized MSE

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_raw = float(loss.item())
        total_raw += loss_raw * xb.size(0)
        sps = xb.size(0) / max(time.time() - t_batch, 1e-6)

        # pretty display
        loss_scaled = loss_raw * loss_scale
        roll_scaled = loss_scaled if roll_scaled is None else (0.9*roll_scaled + 0.1*loss_scaled)
        if use_pbar:
            if log_rmse_m and std_rep is not None:
                rmse_m = torch.sqrt(((err * std_rep) ** 2).mean()).item()
                iterator.set_postfix(loss=f"{roll_scaled:.4f}", rmse_m=f"{rmse_m:.3f} m", sps=f"{sps:.1f}")
            else:
                iterator.set_postfix(loss=f"{roll_scaled:.4f}", sps=f"{sps:.1f}")
        elif (i % log_interval) == 0:
            print(f"[train {epoch:02d}] step {i:05d}/{len(loader)}  "
                  f"loss_scaled={loss_scaled:.4f}  sps={sps:.1f}")

    epoch_time = time.time() - start
    epoch_loss_raw = total_raw / n
    return epoch_loss_raw, epoch_time

@torch.no_grad()
def evaluate(model, loader, device, epoch=0, use_pbar=True,
             stats=None, n_points=None, loss_scale=1.0, log_rmse_m=False,
             optimize_meter_mse=False):
    model.eval()
    total_raw = 0.0
    total_rmse_m = 0.0
    n = len(loader.dataset)

    std_rep = None
    if log_rmse_m or optimize_meter_mse:
        std = torch.tensor(stats["std"], device=device, dtype=torch.float32)
        std_rep = std.repeat(n_points).view(1, 1, -1)

    iterator = loader
    if use_pbar:
        iterator = tqdm(loader, total=len(loader), desc=f"valid {epoch:02d}", leave=False)

    roll_scaled = None
    for xb, yb in iterator:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        err = pred - yb

        if optimize_meter_mse and std_rep is not None:
            loss = ((err * std_rep) ** 2).mean()
        else:
            loss = (err ** 2).mean()

        loss_raw = float(loss.item())
        total_raw += loss_raw * xb.size(0)

        if log_rmse_m and std_rep is not None:
            rmse_m = torch.sqrt(((err * std_rep) ** 2).mean()).item()
            total_rmse_m += rmse_m * xb.size(0)

        if use_pbar:
            loss_scaled = loss_raw * loss_scale
            roll_scaled = loss_scaled if roll_scaled is None else (0.9*roll_scaled + 0.1*loss_scaled)
            if log_rmse_m and std_rep is not None:
                iterator.set_postfix(loss=f"{roll_scaled:.4f}", rmse_m=f"{rmse_m:.3f} m")
            else:
                iterator.set_postfix(loss=f"{roll_scaled:.4f}")

    epoch_loss_raw = total_raw / n
    epoch_rmse_m = (total_rmse_m / n) if (log_rmse_m and std_rep is not None) else None
    return epoch_loss_raw, epoch_rmse_m


# -----------------------------
# Utilities
# -----------------------------
def make_splits(preproc_dir, train_frac=0.9, L_in=8, L_out=1,
                gap=1, cadence_seconds=None, cache_dir="cache",
                train_stride=1, val_stride=None):
    all_paths = sorted(glob.glob(os.path.join(preproc_dir, "*_coords.npy")))
    if not all_paths:
        raise FileNotFoundError(f"No *_coords.npy in {preproc_dir}")
    n_train = max(1, int(len(all_paths) * train_frac))
    train_paths = all_paths[:n_train]
    val_paths   = all_paths[n_train:] if n_train < len(all_paths) else all_paths[-1:]
    if val_stride is None: val_stride = max(1, L_in//2)

    train_ds = ShorelineDataset(train_paths, L_in=L_in, L_out=L_out,
                                stride=train_stride, gap=gap,
                                cadence_seconds=cadence_seconds, cache_dir=cache_dir)
    stats = {"mean": train_ds.mean, "std": train_ds.std}
    val_ds   = ShorelineDataset(val_paths,   L_in=L_in, L_out=L_out,
                                stride=val_stride, gap=gap,
                                stats=stats, cadence_seconds=cadence_seconds, cache_dir=cache_dir)
    return train_ds, val_ds, stats, all_paths


def load_meta(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)

def utm_to_wgs84_line(xy_m, utm_authority):
    # xy_m: [N,2] meters; utm_authority like ["EPSG","32617"]
    utm = CRS.from_authority(*utm_authority)
    wgs84 = CRS.from_epsg(4326)
    inv = Transformer.from_crs(utm, wgs84, always_xy=True)
    lon, lat = inv.transform(xy_m[:,0], xy_m[:,1])
    return LineString(np.stack([lon, lat], axis=1))

# -----------------------------
# CLI: train / predict
# -----------------------------
def cmd_train(args):
    os.makedirs(args.models_dir, exist_ok=True)
    cad_s = parse_cadence_to_seconds(args.cadence) if args.cadence else None
    train_ds, val_ds, stats, all_paths = make_splits(
        args.preproc_dir, args.train_frac, args.L_in, args.L_out,
        gap=args.gap, cadence_seconds=cad_s, cache_dir=args.cache_dir,
        train_stride=args.train_stride, val_stride=max(1, args.L_in//2)
    )

    arr0 = np.load(all_paths[0], mmap_mode="r")
    N = arr0.shape[1]

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoastlineTransformer(n_points=N, L_out=args.L_out,
                                 d_model=args.d_model, nhead=args.nhead,
                                 num_layers=args.num_layers, d_ff=args.d_ff,
                                 dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # CSV logger
    os.makedirs(args.models_dir, exist_ok=True)
    csv_path = os.path.join(args.models_dir, "train_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","epoch_time_sec","samples_per_sec","best_val"])

    best = float("inf")
    best_path = os.path.join(args.models_dir, "coast_tr_best.pt")

    total_samples = len(train_ds)
    batch_per_epoch = len(train_dl)
    print(f"Starting training for {args.epochs} epoch(s) | train samples={len(train_ds)} | val samples={len(val_ds)}")
    for epoch in range(args.epochs):
        train_loss_raw, epoch_time = train_one_epoch(
            model, train_dl, opt, device,
            epoch=epoch, log_interval=args.log_interval, use_pbar=not args.no_pbar,
            stats=stats, n_points=N, loss_scale=args.loss_scale,
            log_rmse_m=args.log_rmse_m, optimize_meter_mse=args.optimize_meter_mse
        )
        val_loss_raw, val_rmse_m = evaluate(
            model, val_dl, device,
            epoch=epoch, use_pbar=not args.no_pbar,
            stats=stats, n_points=N, loss_scale=args.loss_scale,
            log_rmse_m=args.log_rmse_m, optimize_meter_mse=args.optimize_meter_mse
        )

        # scaled for visibility/plots (cosmetic)
        train_loss_scaled = train_loss_raw * args.loss_scale
        val_loss_scaled   = val_loss_raw   * args.loss_scale

        sps = len(train_ds) / max(epoch_time, 1e-6)
        is_best = val_loss_raw < best
        if is_best:
            best = val_loss_raw
            torch.save({"model": model.state_dict(),
                        "stats": stats,
                        "cfg": {"N": N, "L_in": args.L_in, "L_out": args.L_out,
                                "d_model": args.d_model, "nhead": args.nhead,
                                "num_layers": args.num_layers, "d_ff": args.d_ff}},
                       best_path)

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.models_dir, f"coast_tr_epoch{epoch+1}.pt")
            torch.save({"model": model.state_dict(),
                        "stats": stats,
                        "cfg": {"N": N, "L_in": args.L_in, "L_out": args.L_out,
                                "d_model": args.d_model, "nhead": args.nhead,
                                "num_layers": args.num_layers, "d_ff": args.d_ff}},
                       ckpt_path)

        # CSV (delete old CSV once to adopt new columns)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch","train_loss_raw","train_loss_scaled",
                                        "val_loss_raw","val_loss_scaled","val_rmse_m",
                                        "epoch_time_sec","samples_per_sec","best_val_raw"])
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch,
                                    f"{train_loss_raw:.8e}", f"{train_loss_scaled:.6f}",
                                    f"{val_loss_raw:.8e}",   f"{val_loss_scaled:.6f}",
                                    (f"{val_rmse_m:.3f}" if val_rmse_m is not None else ""),
                                    f"{epoch_time:.2f}", f"{sps:.2f}", f"{best:.8e}"])

        msg = (f"epoch {epoch:02d} | train {train_loss_scaled:.4f} (scaled) | "
               f"val {val_loss_scaled:.4f} (scaled)")
        if val_rmse_m is not None:
            msg += f" | val RMSE {val_rmse_m:.3f} m ({val_rmse_m*100:.1f} cm)"
        msg += f" | time {epoch_time:.1f}s | ~{sps:.1f} samp/s | best_raw {best:.2e}"
        print(msg)


@torch.no_grad()
def cmd_predict(args):
    # load ckpt
    ck = torch.load(args.ckpt, map_location="cpu")
    stats = ck["stats"]
    cfg = ck["cfg"]
    N = cfg["N"]
    L_in = cfg["L_in"]
    L_out = cfg["L_out"]

    # load data
    arr = np.load(args.npy)                       # [T,N,2]
    if arr.shape[1] != N or arr.shape[0] < L_in:
        raise ValueError(f"Array shape mismatch or too short. got {arr.shape}, need N={N}, L_in={L_in}")

    x = arr[-L_in:]                               # [L_in,N,2]
    x_norm = (x - stats["mean"]) / stats["std"]
    xb = torch.tensor(x_norm, dtype=torch.float32).view(1, L_in, -1)

    # rebuild model
    model = CoastlineTransformer(n_points=N, L_out=L_out,
                                 d_model=cfg["d_model"], nhead=cfg["nhead"],
                                 num_layers=cfg["num_layers"], d_ff=cfg["d_ff"])
    model.load_state_dict(ck["model"])
    model.eval()

    delta_norm = model(xb)[0, 0].view(N, 2).numpy()
    delta = delta_norm * stats["std"]
    next_xy = x[-1] + delta                        # [N,2] meters (UTM)

    # save as GeoPackage line (WGS84) if meta provided
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.npy))[0].replace("_coords", "")
    if args.meta and os.path.exists(args.meta):
        meta = load_meta(args.meta)
        line_ll = utm_to_wgs84_line(next_xy, meta["utm"])
        gdf = gpd.GeoDataFrame({"segment":[meta.get("segment", base)],
                                "predicted":[True]},
                               geometry=[line_ll], crs="EPSG:4326")
        out_gpkg = os.path.join(args.out_dir, f"{base}_pred.gpkg")
        gdf.to_file(out_gpkg, driver="GPKG")
        print(f"Saved predicted shoreline to {out_gpkg}")
    else:
        out_npy = os.path.join(args.out_dir, f"{base}_pred_xy.npy")
        np.save(out_npy, next_xy)
        print(f"Saved predicted XY (UTM) to {out_npy}")

def build_argparser():
    p = argparse.ArgumentParser(description="Train/predict coastline Transformer")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--preproc_dir", default="preproc")
    pt.add_argument("--models_dir", default="models")
    pt.add_argument("--train_frac", type=float, default=0.9)
    pt.add_argument("--L_in", type=int, default=8)
    pt.add_argument("--L_out", type=int, default=1)
    pt.add_argument("--batch_size", type=int, default=32)
    pt.add_argument("--epochs", type=int, default=18)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=1e-2)
    pt.add_argument("--d_model", type=int, default=256)
    pt.add_argument("--nhead", type=int, default=8)
    pt.add_argument("--num_layers", type=int, default=8)
    pt.add_argument("--d_ff", type=int, default=512)
    pt.add_argument("--dropout", type=float, default=0.11)
    pt.add_argument("--log_interval", type=int, default=10, help="print every N steps when no progress bar")
    pt.add_argument("--save_every", type=int, default=5, help="save a numbered checkpoint every N epochs")
    pt.add_argument("--no_pbar", action="store_true", help="disable tqdm progress bars")
    pt.add_argument("--cadence", default="30D", help="uniform step, e.g. '30D', '7D', '1H', or seconds")
    pt.add_argument("--cache_dir", default="cache", help="where resampled arrays are stored")
    pt.add_argument("--gap", type=int, default=1, help="predict t+gap *steps* after resampling")
    pt.add_argument("--train_stride", type=int, default=12, help="advance windows by this many steps in training")
    pt.add_argument("--loss_scale", type=float, default=100.0,
                    help="multiply displayed/CSV loss by this factor (cosmetic only)")
    pt.add_argument("--log_rmse_m", action="store_true",
                    help="also log RMSE in meters and centimeters")
    pt.add_argument("--optimize_meter_mse", action="store_true",
                    help="optimize MSE in meters instead of normalized units")

    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict")
    pp.add_argument("--npy", required=True, help="preproc\\segment_XX_coords.npy")
    pp.add_argument("--meta", help="preproc\\segment_XX_meta.json")
    pp.add_argument("--ckpt", default=os.path.join("models","coast_tr_best.pt"))
    pp.add_argument("--out_dir", default="output_segments")
    pp.set_defaults(func=cmd_predict)

    return p

def main():
    args = build_argparser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
