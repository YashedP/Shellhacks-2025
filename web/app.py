from flask import Flask, render_template, jsonify, request
import requests
import os

app = Flask(__name__)

# API configuration
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:5001')

def fetch_data_from_api(bounds, year):
    """Fetch data from backend API"""
    try:
        if bounds:
            min_lat, min_lon, max_lat, max_lon = bounds
            url = f"{BACKEND_API_URL}/instance"
            params = {
                'minLat': min_lat,
                'minLon': min_lon,
                'maxLat': max_lat,
                'maxLon': max_lon,
                'year': year
            }
        else:
            # Use global bounds if no bounds specified
            url = f"{BACKEND_API_URL}/instance"
            params = {
                'minLat': -90,
                'minLon': -180,
                'maxLat': 90,
                'maxLon': 180,
                'year': year
            }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return {'points': []}

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
    
    # Fetch data from backend API
    api_data = fetch_data_from_api(bounds, year)
    
    return jsonify({
        'points': api_data.get('points', []),
        'year': year,
        'data_type': 'historical' if year <= 2024 else 'predicted'
    })

@app.route('/api/init_data')
def get_initial_data():
    """Get initial data for the map"""
    # Fetch data for current year (2025) with global bounds
    api_data = fetch_data_from_api(None, 2025)
    
    return jsonify({
        'points': api_data.get('points', []),
        'year': 2025,
        'data_type': 'predicted'
    })

if __name__ == '__main__':
    print(f"Starting web server, connecting to backend API at: {BACKEND_API_URL}")
    app.run(debug=True, host='0.0.0.0', port=5000)