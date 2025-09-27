from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

app = Flask(__name__)

def get_bounding_box_params():
    tlCornerLat = request.args.get('tlCornerLat', default=90.0, type=float)
    tlCornerLon = request.args.get('tlCornerLon', default=-180.0, type=float)
    brCornerLat = request.args.get('brCornerLat', default=-90.0, type=float)
    brCornerLon = request.args.get('brCornerLon', default=180.0, type=float)

    return tlCornerLat, tlCornerLon, brCornerLat, brCornerLon

@app.route("/instant")
def instant():
    # Location
    tlCornerLat, tlCornerLon, brCornerLat, brCornerLon = get_bounding_box_params()

    # Time
    month = request.args.get('month', default=1, type=int)
    year = request.args.get('year', default=1990, type=int)

    return f"{tlCornerLat=} {tlCornerLon=} {brCornerLat=} {brCornerLon=} {month=} {year=}"
CORS(app)  # Enable CORS for all routes

# Global variables for data storage
historical_data = None
predicted_data = None

def load_data():
    """Load historical and predicted data from CSV files"""
    global historical_data, predicted_data
    
    # Load historical data (1950-2024) from web/data folder
    if os.path.exists('../web/data/historical_data.csv'):
        historical_data = pd.read_csv('../web/data/historical_data.csv')
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        print(f"Loaded historical data: {len(historical_data)} records")
    else:
        print("ERROR: No historical data file found at ../web/data/historical_data.csv")
        historical_data = pd.DataFrame()
    
    # Load predicted data (2025-2100) from web/data folder
    if os.path.exists('../web/data/predicted_data.csv'):
        predicted_data = pd.read_csv('../web/data/predicted_data.csv')
        predicted_data['timestamp'] = pd.to_datetime(predicted_data['timestamp'])
        print(f"Loaded predicted data: {len(predicted_data)} records")
    else:
        print("ERROR: No predicted data file found at ../web/data/predicted_data.csv")
        predicted_data = pd.DataFrame()

def extract_coastline_coordinates(gdf):
    """Extract coastline coordinates from the shapefile"""
    coastline_coords = []
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is not None:
            # Get exterior coordinates of the geometry
            if hasattr(geom, 'exterior'):
                coords = list(geom.exterior.coords)
                coastline_coords.extend(coords)
            elif hasattr(geom, 'geoms'):
                # Handle MultiPolygon
                for poly in geom.geoms:
                    if hasattr(poly, 'exterior'):
                        coords = list(poly.exterior.coords)
                        coastline_coords.extend(coords)
    
    # Remove duplicates and filter to reasonable coordinate ranges
    unique_coords = list(set(coastline_coords))
    filtered_coords = [(lon, lat) for lon, lat in unique_coords 
                      if -180 <= lon <= 180 and -90 <= lat <= 90]
    
    return filtered_coords

def create_historical_coastline_data(coastline_coords):
    """Create historical coastline data (1950-2024) - original positions"""
    data = []
    
    # Generate data for each year from 1950 to 2024
    for year in range(1950, 2025):
        for lon, lat in coastline_coords:
            data.append({
                'lat': lat,
                'lon': lon,
                'timestamp': pd.to_datetime(f'{year}-12-31'),
                'sea_level': 0.0  # Base sea level for historical data
            })
    
    return pd.DataFrame(data)

def create_predicted_coastline_data(coastline_coords):
    """Create predicted coastline data (2025-2100) - shifted right from 2030"""
    data = []
    
    # Generate data for each year from 2025 to 2100
    for year in range(2025, 2101):
        for lon, lat in coastline_coords:
            # Shift coastline to the right (eastward) starting from 2030
            shift_amount = 0.0
            if year >= 2030:
                # Progressive shift: more shift in later years
                shift_amount = (year - 2030) * 0.01  # 0.01 degrees per year
            
            shifted_lon = lon + shift_amount
            
            data.append({
                'lat': lat,
                'lon': shifted_lon,
                'timestamp': pd.to_datetime(f'{year}-12-31'),
                'sea_level': shift_amount * 100  # Convert shift to sea level rise for visualization
            })
    
    return pd.DataFrame(data)

def get_data_for_bounds_and_year(topLeftLat, topLeftLong, bottomRightLat, bottomRightLong, year):
    """Get data points within specified bounds for a given year"""
    print(f"Getting data for year {year}, bounds: ({topLeftLat}, {topLeftLong}) to ({bottomRightLat}, {bottomRightLong})")
    
    # Filter by bounds and year
    if historical_data is not None and not historical_data.empty and year <= 2024:
        print(f"Using historical data, shape: {historical_data.shape}")
        year_data = historical_data[historical_data['timestamp'].dt.year == year]
        print(f"After year filter: {len(year_data)} records")
        
        # Apply bounding box filter
        year_data = year_data[
            (year_data['lat'] >= bottomRightLat) & 
            (year_data['lat'] <= topLeftLat) &
            (year_data['lon'] >= topLeftLong) & 
            (year_data['lon'] <= bottomRightLong)
        ]
        print(f"After bounds filter: {len(year_data)} records")
    elif predicted_data is not None and not predicted_data.empty and year >= 2025:
        print(f"Using predicted data, shape: {predicted_data.shape}")
        year_data = predicted_data[predicted_data['timestamp'].dt.year == year]
        print(f"After year filter: {len(year_data)} records")
        
        # Apply bounding box filter
        year_data = year_data[
            (year_data['lat'] >= bottomRightLat) & 
            (year_data['lat'] <= topLeftLat) &
            (year_data['lon'] >= topLeftLong) & 
            (year_data['lon'] <= bottomRightLong)
        ]
        print(f"After bounds filter: {len(year_data)} records")
    else:
        print("No data available for this year")
        year_data = pd.DataFrame()
    
    print(f"Returning {len(year_data)} records")
    return year_data

def create_contour_lines(data):
    """Create contour lines from data points"""
    if data.empty:
        return []
    
    # Simple contour line generation (in a real app, you'd use proper interpolation)
    contours = []
    
    # Group data by sea level ranges
    sea_level_ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
    
    for min_level, max_level in sea_level_ranges:
        level_data = data[(data['sea_level'] >= min_level) & (data['sea_level'] < max_level)]
        
        if not level_data.empty:
            # Create simple contour by connecting nearby points
            points = level_data[['lat', 'lon']].values
            contours.append({
                'level': f"{min_level}-{max_level}m",
                'points': points.tolist(),
                'color': 'red'  # Make all contours red as requested
            })
    
    return contours

@app.route("/instance")
def instance():
    """
    Get a single instance for a bounding box and year
    Parameters: minLat, minLon, maxLat, maxLon, year
    """
    # Get required parameters
    minLat = request.args.get('minLat', type=float)
    minLon = request.args.get('minLon', type=float)
    maxLat = request.args.get('maxLat', type=float)
    maxLon = request.args.get('maxLon', type=float)
    year = request.args.get('year', type=int)
    
    # Validate required parameters
    if any(param is None for param in [minLat, minLon, maxLat, maxLon, year]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Convert min/max bounds to topLeft/bottomRight format for internal processing
    topLeftLat = maxLat
    topLeftLong = minLon
    bottomRightLat = minLat
    bottomRightLong = maxLon
    
    # Get data for the specified bounds and year
    data = get_data_for_bounds_and_year(topLeftLat, topLeftLong, bottomRightLat, bottomRightLong, year)
    
    # Convert to the expected format: array of [lat, lon] pairs
    points = []
    if not data.empty:
        points = data[['lat', 'lon']].values.tolist()
    
    # Create contour lines
    contours = create_contour_lines(data)
    
    return jsonify({
        'points': points,
        'contours': contours
    })

@app.route("/sequence")
def sequence():
    """
    Get a sequence of instances across a range of years
    Parameters: minLat, minLon, maxLat, maxLon, startYear, endYear
    """
    # Get required parameters
    minLat = request.args.get('minLat', type=float)
    minLon = request.args.get('minLon', type=float)
    maxLat = request.args.get('maxLat', type=float)
    maxLon = request.args.get('maxLon', type=float)
    startYear = request.args.get('startYear', type=int)
    endYear = request.args.get('endYear', type=int)
    
    # Validate required parameters
    if any(param is None for param in [minLat, minLon, maxLat, maxLon, startYear, endYear]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Convert min/max bounds to topLeft/bottomRight format for internal processing
    topLeftLat = maxLat
    topLeftLong = minLon
    bottomRightLat = minLat
    bottomRightLong = maxLon
    
    # Get data for each year in the range
    result = {}
    for year in range(startYear, endYear + 1):
        data = get_data_for_bounds_and_year(topLeftLat, topLeftLong, bottomRightLat, bottomRightLong, year)
        
        # Convert to the expected format: array of [lat, lon] pairs
        points = []
        if not data.empty:
            points = data[['lat', 'lon']].values.tolist()
        
        result[str(year)] = points
    
    return jsonify(result)

if __name__ == '__main__':
    # Load data from uploaded files only
    load_data()
    
    # Check if data was loaded successfully
    if historical_data.empty:
        print("WARNING: No historical data loaded!")
    if predicted_data.empty:
        print("WARNING: No predicted data loaded!")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
