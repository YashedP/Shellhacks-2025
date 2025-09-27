from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import folium
from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)

# Global variables for data storage
historical_data = None
predicted_data = None
current_bounds = None

def load_data():
    """Load historical and predicted data from CSV files"""
    global historical_data, predicted_data
    
    # Load historical data (1950-2024) from web/data folder
    if os.path.exists('data/historical_data.csv'):
        historical_data = pd.read_csv('data/historical_data.csv')
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        print(f"Loaded historical data: {len(historical_data)} records")
        print(f"Historical data columns: {list(historical_data.columns)}")
    else:
        print("ERROR: No historical data file found at data/historical_data.csv")
        historical_data = pd.DataFrame()
    
    # Load predicted data (2025-2100) from web/data folder
    if os.path.exists('data/predicted_data.csv'):
        predicted_data = pd.read_csv('data/predicted_data.csv')
        predicted_data['timestamp'] = pd.to_datetime(predicted_data['timestamp'])
        print(f"Loaded predicted data: {len(predicted_data)} records")
        print(f"Predicted data columns: {list(predicted_data.columns)}")
    else:
        print("ERROR: No predicted data file found at data/predicted_data.csv")
        predicted_data = pd.DataFrame()

# Data generation functions removed - using only uploaded data files

def get_data_for_bounds(bounds, year):
    """Get data points within specified bounds for a given year"""
    min_lat, min_lon, max_lat, max_lon = bounds if bounds else (None, None, None, None)
    
    print(f"Getting data for year {year}, bounds: {bounds}")
    
    # Filter by bounds
    if historical_data is not None and year <= 2024:
        print(f"Using historical data, shape: {historical_data.shape}")
        year_data = historical_data[historical_data['timestamp'].dt.year == year]
        print(f"After year filter: {len(year_data)} records")
        if bounds is not None:
            year_data = year_data[
                (year_data['lat'] >= min_lat) & 
                (year_data['lat'] <= max_lat) &
                (year_data['lon'] >= min_lon) & 
                (year_data['lon'] <= max_lon)
            ]
            print(f"After bounds filter: {len(year_data)} records")
    elif predicted_data is not None and year >= 2025:
        print(f"Using predicted data, shape: {predicted_data.shape}")
        year_data = predicted_data[predicted_data['timestamp'].dt.year == year]
        print(f"After year filter: {len(year_data)} records")
        if bounds is not None:
            year_data = year_data[
                (year_data['lat'] >= min_lat) & 
                (year_data['lat'] <= max_lat) &
                (year_data['lon'] >= min_lon) & 
                (year_data['lon'] <= max_lon)
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
                'color': get_contour_color(max_level)
            })
    
    return contours

def get_contour_color(level):
    """Get color for contour based on sea level"""
    if level < 0.5:
        return 'green'
    elif level < 1.0:
        return 'yellow'
    elif level < 1.5:
        return 'orange'
    elif level < 2.0:
        return 'red'
    else:
        return 'darkred'

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/data/<int:year>')
def get_data_for_year(year):
    """Get data for a specific year within current bounds"""
    bounds = request.args.get('bounds')
    if bounds and bounds.strip():  # Check if bounds is not empty or just whitespace
        bounds = [float(x) for x in bounds.split(',')]
    else:
        bounds = None  # Set to None if empty or whitespace
    
    data = get_data_for_bounds(bounds, year)
    
    if data.empty:
        return jsonify({'contours': [], 'points': []})
    
    # Create contour lines
    contours = create_contour_lines(data)
    
    # Get individual points for visualization
    points = data[['lat', 'lon', 'sea_level']].to_dict('records')
    
    return jsonify({
        'contours': contours,
        'points': points,
        'year': year,
        'data_type': 'historical' if year <= 2024 else 'predicted'
    })

@app.route('/api/bounds', methods=['POST'])
def update_bounds():
    """Update the current bounding box for data fragmentation"""
    global current_bounds
    data = request.get_json()
    current_bounds = data.get('bounds')
    return jsonify({'status': 'success'})

@app.route('/api/init_data')
def get_initial_data():
    """Get initial data for the map"""
    # Return data for current year (2024) with default bounds
    data = get_data_for_bounds(None, 2024)
    contours = create_contour_lines(data)
    points = data[['lat', 'lon', 'sea_level']].to_dict('records')
    
    return jsonify({
        'contours': contours,
        'points': points,
        'year': 2024,
        'data_type': 'historical'
    })

if __name__ == '__main__':
    # Load data from uploaded files only
    load_data()
    
    # Check if data was loaded successfully
    if historical_data.empty:
        print("WARNING: No historical data loaded!")
    if predicted_data.empty:
        print("WARNING: No predicted data loaded!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)