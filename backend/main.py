"""
Entry point to the backend.
Written by Yash Jani, Khurram Valiyev and Joshua Sheldon
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

from coastline_mgr import CoastlineMgr

coastline_mgr = CoastlineMgr()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route("/")
def instance():
    """
    Get a single instance for a bounding box and year
    Parameters: minLat, minLon, maxLat, maxLon, year
    """
    # Ensure coastline manager exists
    if coastline_mgr is None:
        return jsonify({"error": "Coastline manager has not been initialized!"}), 500

    # Get required parameters
    minLat = request.args.get("minLat", -90, type=float)
    minLon = request.args.get("minLon", -180, type=float)
    maxLat = request.args.get("maxLat", 90, type=float)
    maxLon = request.args.get("maxLon", 180, type=float)
    year = request.args.get("year", type=int)

    # Validate required parameters
    if year is None:
        return jsonify({"error": "Missing required parameters"}), 400

    # Validate parameter types and ranges
    try:
        # Validate latitude bounds (-90 to 90)
        if not (-90 <= minLat <= 90) or not (-90 <= maxLat <= 90):
            return jsonify({"error": "Latitude must be between -90 and 90"}), 400

        # Validate longitude bounds (-180 to 180)
        if not (-180 <= minLon <= 180) or not (-180 <= maxLon <= 180):
            return jsonify({"error": "Longitude must be between -180 and 180"}), 400

        # Validate that min < max for both lat and lon
        if minLat >= maxLat:
            return jsonify({"error": "minLat must be less than maxLat"}), 400
        if minLon >= maxLon:
            return jsonify({"error": "minLon must be less than maxLon"}), 400

        # Validate reasonable year range (e.g., 1900 to 2100)
        if not (1984 <= year <= 2300):
            return jsonify({"error": "Year must be between 1984 and 2300"}), 400

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

    # Convert to query bounds format (min_lon, min_lat, max_lon, max_lat)
    query_bounds = (minLon, minLat, maxLon, maxLat)

    # Find the closest available timestamp
    if not coastline_mgr.sorted_timestamps:
        return jsonify({"error": "No coastline data available"}), 500

    # Find the closest timestamp to the requested year
    closest_timestamp = min(
        coastline_mgr.sorted_timestamps, key=lambda t: abs((t.year - year))
    )

    # Get coastline points within the query bounds
    points = coastline_mgr.get_coastline_points(closest_timestamp, query_bounds)

    return jsonify({"points": points})


if __name__ == "__main__":
    print("Application ready to serve requests")
    app.run(debug=False, host="0.0.0.0", port=5001)
