# 🌊 Fragmented Coastline Prediction

A hackathon project that visualizes historical and predicted coastline data using an interactive map with time-based controls and zoom-based data fragmentation.

## Features

- **Interactive Map**: Visualize coastline data on an interactive map using Leaflet
- **Time Slider**: Navigate through time from 1950 to 2100 to see historical and predicted data
- **Data Fragmentation**: Zoom in to see only data points for the specific area (bounding box)
- **Contour Lines**: Visualize coastline data as colored contour lines
- **Point Visualization**: Toggle between contour lines and individual data points
- **Responsive Design**: Works on desktop and mobile devices
- **API-Driven**: All data is fetched from backend API services

## Tech Stack

- **Backend**: Python Flask (Web Server)
- **API Backend**: Python FastAPI (Data Service)
- **Frontend**: HTML, CSS, JavaScript (Leaflet for mapping)
- **Mapping**: Leaflet.js (embedded in HTML)

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend API service** (runs on port 5001)

4. **Run the web application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and go to `http://localhost:5000`

## Data Source

The application fetches all data from the backend API service:
- Historical data (1950-2024) is retrieved via API calls
- Predicted data (2025-2100) is retrieved via API calls
- No local data files are required

## How It Works

1. **API Integration**: The Flask web server acts as a proxy to the backend API
2. **Map Display**: The frontend displays an interactive map using Leaflet
3. **Time Navigation**: Use the slider to navigate through different years
4. **Data Fragmentation**: When you zoom in, only data points within the visible area are loaded
5. **Visualization**: Data is displayed as contour lines or individual points based on your selection

## API Endpoints

- `GET /`: Main application page
- `GET /api/data/<year>`: Get data for a specific year within current bounds (proxies to backend API)
- `GET /api/init_data`: Get initial data for the map (proxies to backend API)

The web server acts as a proxy to the backend API service running on port 5001.

## Controls

- **Year Slider**: Navigate through time (1950-2100)
- **Reset View**: Return to the default map view
- **Show Contours**: Toggle contour line visualization
- **Show Points**: Toggle individual data point visualization


## Customization

To use your own data:
1. Update the backend API service to provide your data
2. Ensure your API returns data in the expected format
3. Update the `BACKEND_API_URL` environment variable if needed

## Future Enhancements

- Real-time data updates
- More sophisticated contour line generation
- Data export functionality
- Multiple data source support
- Advanced filtering options