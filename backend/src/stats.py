import geopandas as gpd
import numpy as np
import pandas as pd

# get stats about average change in lat over time from rollout predictions vs real data
def compute_lat_change_stats(gpkg_path, date_col="Date", seg_col="Segment", real_col="geometry"):
    gdf = gpd.read_file(gpkg_path)  # Read the GeoPackage file
    if gdf.empty:
        raise ValueError("The GeoDataFrame is empty. Please check the input file.")
    if date_col not in gdf.columns or seg_col not in gdf.columns or real_col not in gdf.columns:
        raise ValueError(f"One or more required columns are missing in the GeoDataFrame. Required columns: {date_col}, {seg_col}, {real_col}")
    gdf[date_col] = pd.to_datetime(gdf[date_col])  # Ensure date column is datetime
    gdf = gdf.sort_values(by=[seg_col, date_col])  # Sort by segment and date
    lat_changes = []
    for segment, group in gdf.groupby(seg_col):
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            prev_real = group.loc[i-1, real_col]
            curr_real = group.loc[i, real_col]
            if prev_real is None or curr_real is None:
                continue
            real_lat_change = curr_real.centroid.y - prev_real.centroid.y
            lat_changes.append(np.abs(real_lat_change))
    lat_changes = np.array(lat_changes)
    if lat_changes.size == 0:
        raise ValueError("No valid latitude changes were computed. Please check the geometries in the input file.")
    real_changes = lat_changes
    stats = {
        "real_mean": np.mean(real_changes),
        "real_std": np.std(real_changes),
    }
    # print("Latitude Change Stats:", stats)  # Commented out to avoid cluttering output
    return stats

if __name__ == "__main__":
    segment = 1
    stats_list = []
    # average all 250 segments from rollout data stats
    for segment in range(1, 251):
        gpkg_path = f"predictions/segment_{segment}_rollout.gpkg"
        try:
            stats = compute_lat_change_stats(gpkg_path, date_col="step", seg_col="segment")
        except Exception as e:
            print(f"Error processing segment {segment}: {e}")
        stats_list.append(stats)
    # aggregate stats across segments as needed
    # e.g., store in a list and compute overall mean/std
    all_real_means = [s["real_mean"] for s in stats_list if "real_mean" in s]
    all_real_stds = [s["real_std"] for s in stats_list if "real_std" in s]

    overall_stats = {
        "overall_real_mean": np.mean(all_real_means),
        "overall_real_std": np.mean(all_real_stds),
    }
    # print("Overall Latitude Change Stats Across rollout Segments:", overall_stats)  # Commented out to avoid cluttering output
