# 🌊 Fragmented Sea Level Rising Prediction

A hackathon project that visualizes historical and predicted sea level data using an interactive map with time-based controls and zoom-based data fragmentation.

## Features

- **Interactive Map**: Visualize sea level data on an interactive map using Leaflet
- **Time Slider**: Navigate through time from 1950 to 2100 to see historical and predicted data
- **Data Fragmentation**: Zoom in to see only data points for the specific area (bounding box)
- **Contour Lines**: Visualize sea level data as colored contour lines
- **Point Visualization**: Toggle between contour lines and individual data points
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML, CSS, JavaScript (Leaflet for mapping)
- **Data Processing**: Pandas, NumPy
- **Mapping**: Leaflet.js (embedded in HTML)

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:5000`

## Data Format

The application expects CSV files with the following columns:
- `lat`: Latitude coordinate
- `lon`: Longitude coordinate  
- `timestamp`: Date/time of the measurement
- `sea_level`: Sea level measurement in meters

## Sample Data

The application will automatically generate sample data if the CSV files don't exist:
- `data/historical_data.csv`: Historical data from 1950-2024
- `data/predicted_data.csv`: Predicted data from 2025-2100

## How It Works

1. **Data Loading**: The Flask backend loads historical and predicted data from CSV files
2. **Map Display**: The frontend displays an interactive map using Leaflet
3. **Time Navigation**: Use the slider to navigate through different years
4. **Data Fragmentation**: When you zoom in, only data points within the visible area are loaded
5. **Visualization**: Data is displayed as contour lines or individual points based on your selection

## API Endpoints

- `GET /`: Main application page
- `GET /api/data/<year>`: Get data for a specific year within current bounds
- `POST /api/bounds`: Update the current bounding box for data fragmentation
- `GET /api/init_data`: Get initial data for the map

## Controls

- **Year Slider**: Navigate through time (1950-2100)
- **Reset View**: Return to the default map view
- **Show Contours**: Toggle contour line visualization
- **Show Points**: Toggle individual data point visualization

## Legend

- 🟢 Green: 0 - 0.5m sea level
- 🟡 Yellow: 0.5 - 1.0m sea level  
- 🟠 Orange: 1.0 - 1.5m sea level
- 🔴 Red: 1.5 - 2.0m sea level
- 🟤 Dark Red: 2.0m+ sea level

## Customization

To use your own data:
1. Replace the sample CSV files in the `data/` directory
2. Ensure your data follows the required format (lat, lon, timestamp, sea_level)
3. Restart the application

## Future Enhancements

- Real-time data updates
- More sophisticated contour line generation
- Data export functionality
- Multiple data source support
- Advanced filtering options