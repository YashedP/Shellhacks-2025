from flask import Flask, request, jsonify
from flask_cors import CORS
from coastline_mgr import CoastlineMgr

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    
    # Validate parameter types and ranges
    try:
        # Validate latitude bounds (-90 to 90)
        if not (-90 <= minLat <= 90) or not (-90 <= maxLat <= 90):
            return jsonify({'error': 'Latitude must be between -90 and 90'}), 400
        
        # Validate longitude bounds (-180 to 180)
        if not (-180 <= minLon <= 180) or not (-180 <= maxLon <= 180):
            return jsonify({'error': 'Longitude must be between -180 and 180'}), 400
        
        # Validate that min < max for both lat and lon
        if minLat >= maxLat:
            return jsonify({'error': 'minLat must be less than maxLat'}), 400
        if minLon >= maxLon:
            return jsonify({'error': 'minLon must be less than maxLon'}), 400
        
        # Validate reasonable year range (e.g., 1900 to 2100)
        if not (1900 <= year <= 2100):
            return jsonify({'error': 'Year must be between 1900 and 2100'}), 400
            
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameter type: {str(e)}'}), 400
    
    # Convert min/max bounds to topLeft/bottomRight format for internal processing
    topLeftLat = maxLat
    topLeftLong = minLon
    bottomRightLat = minLat
    bottomRightLong = maxLon
    
    # TODO: Implement data fetching logic
    # For now, return empty data structure
    return jsonify({
        'points': [],
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
    
    # Validate parameter types and ranges
    try:
        # Validate latitude bounds (-90 to 90)
        if not (-90 <= minLat <= 90) or not (-90 <= maxLat <= 90):
            return jsonify({'error': 'Latitude must be between -90 and 90'}), 400
        
        # Validate longitude bounds (-180 to 180)
        if not (-180 <= minLon <= 180) or not (-180 <= maxLon <= 180):
            return jsonify({'error': 'Longitude must be between -180 and 180'}), 400
        
        # Validate that min < max for both lat and lon
        if minLat >= maxLat:
            return jsonify({'error': 'minLat must be less than maxLat'}), 400
        if minLon >= maxLon:
            return jsonify({'error': 'minLon must be less than maxLon'}), 400
        
        # Validate year range
        if startYear > endYear:
            return jsonify({'error': 'startYear must be less than or equal to endYear'}), 400
        
        # Validate reasonable year range (e.g., 1900 to 2100)
        if not (1900 <= startYear <= 2100) or not (1900 <= endYear <= 2100):
            return jsonify({'error': 'Years must be between 1900 and 2100'}), 400
            
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameter type: {str(e)}'}), 400
    
    # Convert min/max bounds to topLeft/bottomRight format for internal processing
    topLeftLat = maxLat
    topLeftLong = minLon
    bottomRightLat = minLat
    bottomRightLong = maxLon
    
    # TODO: Implement data fetching logic
    # For now, return empty data structure
    result = {}
    for year in range(startYear, endYear + 1):
        result[str(year)] = []
    
    return jsonify(result)

if __name__ == '__main__':
    # Initialize function - equivalent to Lambda cold start
    print("Starting application initialization...")

    
    # Check if data was loaded successfully
    if historical_data is None or historical_data.empty:
        print("WARNING: No historical data loaded!")
    if predicted_data is None or predicted_data.empty:
        print("WARNING: No predicted data loaded!")
    
    print("Application ready to serve requests")
    app.run(debug=True, host='0.0.0.0', port=5001)
