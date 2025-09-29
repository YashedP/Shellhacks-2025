"""
Entry point to the backend.
Written by Khurram Valiyev
"""

from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

from coastline_mgr import CoastlineMgr

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global CoastlineMgr instance
coastline_mgr = None


@app.route("/instance")
def instance():
    """
    Get a single instance for a bounding box and year
    Parameters: minLat, minLon, maxLat, maxLon, year
    """
    # Get required parameters
    minLat = request.args.get("minLat", type=float)
    minLon = request.args.get("minLon", type=float)
    maxLat = request.args.get("maxLat", type=float)
    maxLon = request.args.get("maxLon", type=float)
    year = request.args.get("year", type=int)

    # Validate required parameters
    if any(param is None for param in [minLat, minLon, maxLat, maxLon, year]):
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

    # Create datetime object for the year (using January 1st)
    target_date = datetime(year, 1, 1)

    # Find the closest available timestamp
    available_timestamps = list(coastline_mgr.coastlines.keys())
    if not available_timestamps:
        return jsonify({"error": "No coastline data available"}), 404

    # Find the closest timestamp to the requested year
    closest_timestamp = min(available_timestamps, key=lambda t: abs((t.year - year)))

    # Get coastline points within the query bounds
    points = coastline_mgr.get_coastline_points(closest_timestamp, query_bounds)

    return jsonify({"points": points})


@app.route("/sequence")
def sequence():
    """
    Get a sequence of instances across a range of years
    Parameters: minLat, minLon, maxLat, maxLon, startYear, endYear
    """
    # Get required parameters
    minLat = request.args.get("minLat", type=float)
    minLon = request.args.get("minLon", type=float)
    maxLat = request.args.get("maxLat", type=float)
    maxLon = request.args.get("maxLon", type=float)
    startYear = request.args.get("startYear", type=int)
    endYear = request.args.get("endYear", type=int)

    # Validate required parameters
    if any(
        param is None for param in [minLat, minLon, maxLat, maxLon, startYear, endYear]
    ):
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

        # Validate year range
        if startYear > endYear:
            return (
                jsonify({"error": "startYear must be less than or equal to endYear"}),
                400,
            )

        # Validate reasonable year range (e.g., 1900 to 2100)
        if not (1984 <= startYear <= 2300) or not (1984 <= endYear <= 2300):
            return jsonify({"error": "Years must be between 1900 and 2100"}), 400

        if startYear == endYear:
            return jsonify({"error": "startYear and endYear must be different"}), 400

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

    # Convert to query bounds format (min_lon, min_lat, max_lon, max_lat)
    query_bounds = (minLon, minLat, maxLon, maxLat)

    available_timestamps = list(coastline_mgr.coastlines.keys())
    result = {}
    for year in range(startYear, endYear + 1):
        # Find the closest timestamp to the requested year
        closest_timestamp = min(
            available_timestamps, key=lambda t: abs((t.year - year))
        )

        # Get coastline points within the query bounds
        points = coastline_mgr.get_coastline_points(closest_timestamp, query_bounds)

        result[str(year)] = points
    return jsonify(result)


if __name__ == "__main__":
    # Initialize function - equivalent to Lambda cold start
    print("Starting application initialization...")

    coastline_mgr = CoastlineMgr()

    print("Application ready to serve requests")
    app.run(debug=False, host="0.0.0.0", port=5001)
