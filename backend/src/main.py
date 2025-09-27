from flask import Flask, request

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

@app.route("/sequence")
def sequence():
    # Location
    tlCornerLat, tlCornerLon, brCornerLat, brCornerLon = get_bounding_box_params()

    # Time
    startMonth = request.args.get('startMonth', default=1, type=int)
    startYear = request.args.get('startYear', default=1990, type=int)
    endMonth = request.args.get('endMonth', default=1, type=int)
    endYear = request.args.get('endYear', default=1990, type=int)

    return f"{tlCornerLat=} {tlCornerLon=} {brCornerLat=} {brCornerLon=} {startMonth=} {startYear=} {endMonth=} {endYear=}"
