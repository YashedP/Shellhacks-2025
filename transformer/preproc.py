# preproc.py
import os, sys, glob, json, math, argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import linemerge
from shapely.geometry import LineString
from pyproj import CRS

# ---------- helpers ----------
def pick_utm(lon, lat):
    zone = int((lon + 180)//6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def merge_to_line(geom):
    # Accept LineString or MultiLineString; pick the longest part if needed
    if geom is None:
        return None
    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "MultiLineString":
            merged = max(merged.geoms, key=lambda g: g.length)
        return merged
    return geom

def force_2d(geom):
    if geom is None:
        return None
    # Drop Z if present; robust for LineString/MultiLineString
    if hasattr(geom, "has_z") and geom.has_z:
        if geom.geom_type == "LineString":
            return LineString([(x, y) for (x, y, *rest) in geom.coords])
        elif geom.geom_type == "MultiLineString":
            parts = [LineString([(x, y) for (x, y, *rest) in g.coords]) for g in geom.geoms]
            return linemerge(parts)
    return geom

def sample_line(line, N):
    if line is None or line.is_empty:
        return None
    L = line.length
    if L == 0 or not np.isfinite(L):
        p = np.array(line.coords[:1]).repeat(N, axis=0)
        return p[:, :2]
    d = np.linspace(0, L, N)
    pts = [line.interpolate(di) for di in d]
    return np.array([[p.x, p.y] for p in pts], dtype=np.float32)  # [N,2]

def load_gpkg(path, layer=None):
    try:
        return gpd.read_file(path, layer=layer)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path} ({'layer='+layer if layer else 'default layer'}): {e}")

# ---------- core preprocessing ----------
def preprocess_gpkg(gpkg_path, out_dir="preproc", layer=None, date_col="Date", seg_col="Segment", N=128):
    os.makedirs(out_dir, exist_ok=True)
    gdf = load_gpkg(gpkg_path, layer=layer)

    # Basic checks
    if "geometry" not in gdf.columns:
        raise ValueError(f"{gpkg_path}: no geometry column")
    if date_col not in gdf.columns:
        raise ValueError(f"{gpkg_path}: missing '{date_col}' column")
    if seg_col not in gdf.columns:
        # not fatal; we can still continue
        print(f"[WARN] {gpkg_path}: missing '{seg_col}' column; using filename as segment id")

    # Ensure datetime & sort
    gdf[date_col] = pd.to_datetime(gdf[date_col], errors="coerce")
    gdf = gdf.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if gdf.empty:
        raise ValueError(f"{gpkg_path}: no rows with valid {date_col}")

    # Force 2D + merge into one line per row
    gdf["geometry"] = gdf["geometry"].apply(force_2d).apply(merge_to_line)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    if gdf.empty:
        raise ValueError(f"{gpkg_path}: geometries empty after 2D+merge")

    # CRS handling
    if gdf.crs is None:
        # assume WGS84 if truly missing (adjust if you know better)
        print(f"[WARN] {gpkg_path}: no CRS set; assuming EPSG:4326 (lon/lat).")
        gdf.set_crs("EPSG:4326", inplace=True)

    # Choose UTM based on first geometry centroid in lon/lat
    gdf_ll = gdf.to_crs(4326)
    centroid = gdf_ll.geometry.iloc[0].centroid
    utm = pick_utm(centroid.x, centroid.y)

    # Reproject to meters
    gdf_m = gdf.to_crs(utm)

    # Sample each shoreline to N points
    X, dates = [], []
    for _, row in gdf_m.iterrows():
        xy = sample_line(row.geometry, N)
        if xy is None:
            continue
        X.append(xy)
        dates.append(pd.to_datetime(row[date_col]).isoformat())

    if not X:
        raise ValueError(f"{gpkg_path}: no valid sampled polylines")

    X = np.stack(X, axis=0)  # [T,N,2]
    seg_id = str(gdf[seg_col].iloc[0]) if seg_col in gdf.columns else os.path.splitext(os.path.basename(gpkg_path))[0]
    base = os.path.splitext(os.path.basename(gpkg_path))[0]

    # Save
    np.save(os.path.join(out_dir, f"{base}_coords.npy"), X)
    with open(os.path.join(out_dir, f"{base}_meta.json"), "w") as f:
        json.dump({"segment": seg_id, "dates": dates, "utm": utm.to_authority()}, f)

    print(f"[OK] {base}: T={X.shape[0]} N={X.shape[1]} → {out_dir}\\{base}_coords.npy")
    return True

# ---------- batch runner / CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Preprocess shoreline GeoPackages to fixed arrays")
    ap.add_argument("--input_dir", default="output_segments", help="Folder containing .gpkg files (can be output_segments if that’s your source)")
    ap.add_argument("--pattern", default="*.gpkg", help="Glob pattern for files, e.g., '*.gpkg' or 'segment_*.gpkg'")
    ap.add_argument("--out_dir", default="preproc", help="Output folder for *_coords.npy and *_meta.json")
    ap.add_argument("--layer", default=None, help="Layer name if the gpkg has multiple layers")
    ap.add_argument("--date_col", default="Date", help="Column with timestamps")
    ap.add_argument("--seg_col", default="Segment", help="Column with segment id")
    ap.add_argument("--N", type=int, default=128, help="Samples along shoreline")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    args = ap.parse_args()

    # Gather files
    if args.recursive:
        files = [p for p in glob.glob(os.path.join(args.input_dir, "**", args.pattern), recursive=True)]
    else:
        files = [p for p in glob.glob(os.path.join(args.input_dir, args.pattern))]
    files = sorted(files)

    if not files:
        print(f"[ERR] No files matched: {os.path.join(args.input_dir, '**' if args.recursive else '') , args.pattern}")
        sys.exit(1)

    print(f"Found {len(files)} file(s). Output → {args.out_dir}")
    ok, fail = 0, 0
    for path in files:
        try:
            preprocess_gpkg(
                gpkg_path=path,
                out_dir=args.out_dir,
                layer=args.layer,
                date_col=args.date_col,
                seg_col=args.seg_col,
                N=args.N,
            )
            ok += 1
        except Exception as e:
            print(f"[FAIL] {os.path.basename(path)} → {e}")
            fail += 1

    print(f"\nDone. Success: {ok}  Failed: {fail}")
    if fail > 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
