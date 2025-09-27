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
class ShorelineDataset(Dataset):
    def __init__(self, npy_paths, L_in=8, L_out=1, stride=1, stats=None):
        self.L_in, self.L_out = L_in, L_out
        self.samples = []  # (path, start_idx)
        self.paths = []
        for p in npy_paths:
            T = np.load(p, mmap_mode="r").shape[0]
            if T >= L_in + L_out:
                self.paths.append(p)
                for s in range(0, T - (L_in + L_out) + 1, stride):
                    self.samples.append((p, s))

        if stats is None:
            ssum = np.zeros(2, dtype=np.float64)
            ssum2 = np.zeros(2, dtype=np.float64)
            count = 0
            for p in self.paths:
                arr = np.load(p, mmap_mode="r")  # [T,N,2]
                flat = arr.reshape(-1, 2)
                ssum  += flat.sum(0)
                ssum2 += (flat**2).sum(0)
                count += flat.shape[0]
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
        y = arr[s+self.L_in:s+self.L_in+self.L_out]        # [L_out,N,2]

        # normalize by (x,y) channels
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # predict deltas wrt last input frame
        last = x[-1]                                       # [N,2]
        y = y - last                                       # [L_out,N,2]

        x = torch.tensor(x, dtype=torch.float32).view(self.L_in, -1)   # [L_in,N*2]
        y = torch.tensor(y, dtype=torch.float32).view(self.L_out, -1)  # [L_out,N*2]
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
def train_one_epoch(model, loader, opt, device, epoch=0, log_interval=10, use_pbar=True):
    model.train()
    total = 0.0
    n = len(loader.dataset)
    start = time.time()
    iterator = enumerate(loader)
    if use_pbar:
        iterator = tqdm(iterator, total=len(loader), desc=f"train {epoch:02d}", leave=False)

    # simple rolling avg to stabilize the live loss display
    roll = None
    for i, (xb, yb) in iterator:
        t_batch = time.time()
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = ((pred - yb) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_val = float(loss.item())
        total += loss_val * xb.size(0)
        sps = xb.size(0) / max(time.time() - t_batch, 1e-6)

        roll = loss_val if roll is None else (0.9 * roll + 0.1 * loss_val)
        if use_pbar:
            iterator.set_postfix(loss=f"{roll:.4f}", sps=f"{sps:.1f}")
        elif (i % log_interval) == 0:
            print(f"[train {epoch:02d}] step {i:05d}/{len(loader)}  loss={loss_val:.4f}  sps={sps:.1f}")

    epoch_time = time.time() - start
    epoch_loss = total / n
    return epoch_loss, epoch_time

@torch.no_grad()
def evaluate(model, loader, device, epoch=0, use_pbar=True):
    model.eval()
    total = 0.0
    n = len(loader.dataset)
    iterator = loader
    if use_pbar:
        iterator = tqdm(loader, total=len(loader), desc=f"valid {epoch:02d}", leave=False)

    roll = None
    for xb, yb in iterator:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = ((pred - yb) ** 2).mean()
        loss_val = float(loss.item())
        total += loss_val * xb.size(0)
        roll = loss_val if roll is None else (0.9 * roll + 0.1 * loss_val)
        if use_pbar:
            iterator.set_postfix(loss=f"{roll:.4f}")

    epoch_loss = total / n
    return epoch_loss

# -----------------------------
# Utilities
# -----------------------------
def make_splits(preproc_dir, train_frac=0.9, L_in=8, L_out=1):
    all_paths = sorted(glob.glob(os.path.join(preproc_dir, "*_coords.npy")))
    if not all_paths:
        raise FileNotFoundError(f"No *_coords.npy in {preproc_dir}")
    n_train = max(1, int(len(all_paths) * train_frac))
    train_paths = all_paths[:n_train]
    val_paths = all_paths[n_train:] if n_train < len(all_paths) else all_paths[-1:]

    train_ds = ShorelineDataset(train_paths, L_in=L_in, L_out=L_out, stride=1)
    stats = {"mean": train_ds.mean, "std": train_ds.std}
    val_ds = ShorelineDataset(val_paths, L_in=L_in, L_out=L_out, stride=max(1, L_in//2), stats=stats)
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
    train_ds, val_ds, stats, all_paths = make_splits(args.preproc_dir, args.train_frac, args.L_in, args.L_out)

    arr0 = np.load(all_paths[0], mmap_mode="r")
    N = arr0.shape[1]

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

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
        train_loss, epoch_time = train_one_epoch(
            model, train_dl, opt, device,
            epoch=epoch, log_interval=args.log_interval, use_pbar=not args.no_pbar
        )
        val_loss = evaluate(model, val_dl, device, epoch=epoch, use_pbar=not args.no_pbar)

        sps = total_samples / max(epoch_time, 1e-6)
        is_best = val_loss < best
        if is_best:
            best = val_loss
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

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                    f"{epoch_time:.2f}", f"{sps:.2f}", f"{best:.6f}"])

        print(f"epoch {epoch:02d} | train {train_loss:.6f} | val {val_loss:.6f} | "
              f"time {epoch_time:.1f}s | ~{sps:.1f} samp/s | best {best:.6f}")

    print(f"Saved best checkpoint -> {best_path}\nCSV log -> {csv_path}")


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
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=1e-2)
    pt.add_argument("--d_model", type=int, default=256)
    pt.add_argument("--nhead", type=int, default=8)
    pt.add_argument("--num_layers", type=int, default=4)
    pt.add_argument("--d_ff", type=int, default=512)
    pt.add_argument("--dropout", type=float, default=0.1)
    pt.add_argument("--log_interval", type=int, default=10, help="print every N steps when no progress bar")
    pt.add_argument("--save_every", type=int, default=5, help="save a numbered checkpoint every N epochs")
    pt.add_argument("--no_pbar", action="store_true", help="disable tqdm progress bars")

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
