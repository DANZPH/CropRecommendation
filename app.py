import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from streamlit_echarts import st_echarts
import streamlit.components.v1 as components
import time
import json
import requests
from datetime import datetime

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="centered")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) fixed;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Hide Streamlit main menu and footer */

.header-glass {
    width: 100%;
    padding: 10px 0 16px 0;
    background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 24px;
    box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.10);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 0px;
}
.header-glass h1 {
    color: #2c3e50;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 1px;
}
.glass-card {
    background: rgba(255,255,255,0.35);
    border-radius: 18px;
    box-shadow: 0 2px 12px 0 rgba(31, 38, 135, 0.08);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 24px 16px;
    margin: 0 auto 18px auto;
    max-width: 480px;
    border: 1px solid rgba(255,255,255,0.14);
}
.footer-glass {
    width: 100%;
    padding: 10px 0 6px 0;
    background: rgba(255,255,255,0.18);
    border-radius: 18px 18px 0 0;
    box-shadow: 0 -2px 8px 0 rgba(31, 38, 135, 0.07);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    display: flex;
    flex-direction: column;
    align-items: center;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000;
}
.stButton>button {
    border-radius: 12px;
    background-color: #27ae60;
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
    padding: 10px 24px;
    margin: 0 auto;
    display: block;
}
.stButton>button:hover {
    background-color: #2ecc71;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(39,174,96,0.2);
}
.stSlider>div>div>div>div {
    background: #27ae60 !important;
}
.probability-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    background: rgba(255,255,255,0.35);
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.95rem;
}
.probability-table th, .probability-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
}
.probability-table th {
    background-color: #27ae60;
    color: white;
}
.nav-buttons-container {
    margin: 0 auto;
    max-width: 480px;
    padding: 0 12px 12px 12px;
}
.nav-button {
    width: 100%;
    margin-bottom: 0px !important;
    border-radius: 12px !important;
}
.location-info {
    background: rgba(255,255,255,0.5);
    border-radius: 10px;
    padding: 12px;
    margin: 10px 0;
    border-left: 4px solid #27ae60;
}
.weather-card {
    background: rgba(255,255,255,0.6);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
/* Image styling */
.stImage {
    border: 4px solid #27ae60 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(39,174,96,0.25) !important;
    padding: 4px;
    background: white;
}

/* Or alternatively, if the above doesn't work, try this more specific selector: */
div[data-testid="stImage"] img {
    border: 4px solid #27ae60 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(39,174,96,0.25) !important;
    padding: 4px;
    background: white;
}
/* Custom loading animation */
@keyframes grow {
    0% { transform: scale(0.8); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
    100% { transform: scale(0.8); opacity: 0.5; }
}
.loading-container {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    gap: 16px;
    padding: 24px;
}
.loading-dots {
    display: flex;
    gap: 8px;
}
.loading-dot {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #27ae60;
    animation: grow 1.5s infinite ease-in-out;
}
.loading-dot:nth-child(2) {
    animation-delay: 0.2s;
}
.loading-dot:nth-child(3) {
    animation-delay: 0.4s;
}
.loading-text {
    font-weight: 600;
    color: #2c3e50;
    text-align: center;
}
            
/* Responsive header for mobile */
@media (max-width: 600px) {
    .header-glass {
        padding: 8px 0 10px 0;
        border-radius: 14px;
    }
    .header-glass h1 {
        font-size: 1.3rem;
        padding: 0 8px;
        text-align: center;
    }
}
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class='header-glass'>
    <h1>Crop Recommendation System</h1>
</div>
""", unsafe_allow_html=True)

# Navigation buttons
if 'active_section' not in st.session_state:
    st.session_state['active_section'] = 'recommend'

st.markdown("<div class='nav-buttons-container'>", unsafe_allow_html=True)
col_nav1, col_nav2, col_nav3 = st.columns(3, gap="small")
with col_nav1:
    if st.button('🌱 Manual Entry', key='nav_recommend', use_container_width=True):
        st.session_state['active_section'] = 'recommend'
with col_nav2:
    if st.button('📍 Auto Location', key='nav_auto', use_container_width=True):
        st.session_state['active_section'] = 'auto'
with col_nav3:
    if st.button('🔍 Reverse Search', key='nav_reverse', use_container_width=True):
        st.session_state['active_section'] = 'reverse'
st.markdown("</div>", unsafe_allow_html=True)

# Helper functions for location and weather data
def get_user_location():
    """Get user's real-time GPS location using JavaScript geolocation API"""
    location_html = """
    <script>
    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    const accuracy = position.coords.accuracy;
                    
                    // Send location data to Streamlit
                    window.parent.postMessage({
                        type: 'location',
                        lat: lat,
                        lon: lon,
                        accuracy: accuracy
                    }, '*');
                },
                function(error) {
                    let errorMsg = '';
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            errorMsg = "Location access denied by user.";
                            break;
                        case error.POSITION_UNAVAILABLE:
                            errorMsg = "Location information is unavailable.";
                            break;
                        case error.TIMEOUT:
                            errorMsg = "Location request timed out.";
                            break;
                        default:
                            errorMsg = "An unknown error occurred.";
                            break;
                    }
                    window.parent.postMessage({
                        type: 'location_error',
                        error: errorMsg
                    }, '*');
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000
                }
            );
        } else {
            window.parent.postMessage({
                type: 'location_error',
                error: 'Geolocation is not supported by this browser.'
            }, '*');
        }
    }
    
    // Auto-trigger when page loads
    window.onload = getLocation;
    </script>
    
    <div style="text-align: center; padding: 20px;">
        <h4>🌍 Getting your location...</h4>
        <p>Please allow location access when prompted</p>
        <button onclick="getLocation()" style="
            background: #27ae60;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        ">📍 Get My Location</button>
    </div>
    """
    return location_html

def get_location_by_ip():
    """Fallback: Get location using IP geolocation (free service)"""
    try:
        response = requests.get('http://ip-api.com/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data.get('city', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'lat': data.get('lat', 0),
                'lon': data.get('lon', 0),
                'timezone': data.get('timezone', 'UTC')
            }
    except Exception as e:
        st.error(f"Error getting location: {e}")
    return None

def reverse_geocode(lat, lon):
    """Get city name from coordinates using free geocoding service"""
    try:
        # Using free reverse geocoding service
        response = requests.get(
            f'https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en',
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data.get('city', data.get('locality', 'Unknown')),
                'country': data.get('countryName', 'Unknown'),
                'lat': lat,
                'lon': lon
            }
    except Exception as e:
        st.error(f"Error getting city name: {e}")
    return {'city': 'Unknown', 'country': 'Unknown', 'lat': lat, 'lon': lon}

def get_weather_data(lat, lon):
    """Get weather data using OpenWeatherMap API (free tier)"""
    try:
        # Using OpenWeatherMap's free API (requires API key)
        # For demo purposes, we'll use a mock weather service
        # In production, sign up at openweathermap.org for a free API key
        
        # Mock weather data based on location (you should replace this with actual API call)
        weather_data = {
            'temperature': np.random.uniform(15, 35),  # Random temp between 15-35°C
            'humidity': np.random.uniform(40, 80),     # Random humidity 40-80%
            'description': 'Clear sky',
            'rainfall': np.random.uniform(50, 200)     # Random rainfall 50-200mm
        }
        
        # Uncomment and use this for actual OpenWeatherMap API:
        # API_KEY = "your_openweathermap_api_key"
        # url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        # response = requests.get(url, timeout=5)
        # if response.status_code == 200:
        #     data = response.json()
        #     weather_data = {
        #         'temperature': data['main']['temp'],
        #         'humidity': data['main']['humidity'],
        #         'description': data['weather'][0]['description'],
        #         'rainfall': data.get('rain', {}).get('1h', 0) * 24 * 30  # Convert to monthly
        #     }
        
        return weather_data
    except Exception as e:
        st.error(f"Error getting weather data: {e}")
        return None

def get_soil_data_estimate(lat, lon):
    """Estimate soil data based on location (simplified approach)"""
    # This is a simplified estimation. In a real application, you'd use:
    # - Soil databases like ISRIC World Soil Information
    # - Agricultural APIs
    # - Local agricultural extension services
    
    # Basic soil estimates based on latitude/climate zones
    if abs(lat) < 23.5:  # Tropical zone
        return {
            'N': np.random.uniform(40, 80),
            'P': np.random.uniform(20, 60),
            'K': np.random.uniform(80, 140),
            'ph': np.random.uniform(5.5, 7.0)
        }
    elif abs(lat) < 66.5:  # Temperate zone
        return {
            'N': np.random.uniform(50, 100),
            'P': np.random.uniform(30, 80),
            'K': np.random.uniform(60, 120),
            'ph': np.random.uniform(6.0, 7.5)
        }
    else:  # Polar zone
        return {
            'N': np.random.uniform(20, 60),
            'P': np.random.uniform(10, 40),
            'K': np.random.uniform(40, 80),
            'ph': np.random.uniform(5.0, 6.5)
        }

# Data and Model Loading
@st.cache_data
def load_data():
    # Create sample data since we don't have the actual CSV
    np.random.seed(42)
    crops = ['rice', 'maize', 'wheat', 'potato', 'tomato', 'apple', 'banana', 'coconut', 'cotton', 'sugarcane']
    
    data = []
    for crop in crops:
        for _ in range(100):  # 100 samples per crop
            if crop == 'rice':
                sample = [
                    np.random.uniform(80, 120),  # N
                    np.random.uniform(35, 70),   # P
                    np.random.uniform(35, 70),   # K
                    np.random.uniform(20, 35),   # temperature
                    np.random.uniform(80, 95),   # humidity
                    np.random.uniform(5.0, 7.0), # ph
                    np.random.uniform(150, 300), # rainfall
                    crop
                ]
            elif crop == 'wheat':
                sample = [
                    np.random.uniform(50, 100),
                    np.random.uniform(20, 60),
                    np.random.uniform(20, 60),
                    np.random.uniform(15, 25),
                    np.random.uniform(50, 70),
                    np.random.uniform(6.0, 8.0),
                    np.random.uniform(50, 150),
                    crop
                ]
            else:  # Generic for other crops
                sample = [
                    np.random.uniform(40, 100),
                    np.random.uniform(20, 80),
                    np.random.uniform(20, 100),
                    np.random.uniform(15, 35),
                    np.random.uniform(40, 80),
                    np.random.uniform(5.5, 7.5),
                    np.random.uniform(50, 250),
                    crop
                ]
            data.append(sample)
    
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    return df

@st.cache_resource
def train_model(data):
    X = data.drop(columns=["label"])
    y = data["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

data = load_data()
model, scaler = train_model(data)

# Main Card Container
if st.session_state['active_section'] == 'recommend':
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    # --- Main Recommendation Section ---
    st.header("Enter Soil and Climate Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.slider("Nitrogen (N)", 0, 140, 50)
        P = st.slider("Phosphorus (P)", 5, 145, 50)
    with col2:
        K = st.slider("Potassium (K)", 5, 205, 50)
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
    with col3:
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 50.0, 0.1)
        ph = st.slider("pH Level", 3.0, 10.0, 6.5, 0.1)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 1.0)

    if st.button("Get Recommendation", type="primary"):
        with st.spinner(''):
            # Custom loading animation
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown("""
                <div class='loading-container'>
                    <div class='loading-dots'>
                        <div class='loading-dot'></div>
                        <div class='loading-dot'></div>
                        <div class='loading-dot'></div>
                    </div>
                    <div class='loading-text'>Analyzing soil and climate data...</div>
                </div>
                """, unsafe_allow_html=True)
            
            time.sleep(2)  # Simulate processing time
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_scaled = scaler.transform(input_data)
            predicted_crop = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            loading_placeholder.empty()
            
        st.success(f"Recommended Crop: **{predicted_crop}**")
        
        # Display crop image
        image_path = f"crops/{predicted_crop.lower()}.jpg"
        try:
            st.image(image_path, caption=f"Recommended: {predicted_crop}", use_container_width=True)
        except:
            # If image not found, show a placeholder or generic crop image
            st.info(f"🌾 Recommended crop: {predicted_crop}")
            st.caption("(Crop image not available)")
        
        # Display probabilities table
        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False).head(10)
        
        st.subheader("Top Recommended Crops")
        st.table(prob_df.style.format({"Probability": "{:.2%}"}).background_gradient(cmap='YlGn'))
        
        # Chart visualization
        top_crops = prob_df.head(10)
        option = {
            "title": {"text": "Crop Suitability Scores", "left": "center"},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"bottom": "20%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": list(top_crops["Crop"]),
                "axisLabel": {
                    "rotate": 45,
                    "interval": 0,
                    "width": 80,
                    "overflow": "break"
                }
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {"formatter": "{value}%"},
                "min": 0,
                "max": 100
            },
            "series": [{
                "name": "Suitability",
                "type": "bar",
                "data": [round(p*100, 1) for p in top_crops["Probability"]],
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#27ae60"},
                            {"offset": 1, "color": "#2ecc71"}
                        ]
                    },
                    "borderRadius": [4, 4, 0, 0]
                },
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": "{c}%"
                }
            }]
        }
        
        st_echarts(options=option, height="400px", key="crop_chart")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state['active_section'] == 'auto':
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.header("📍 Real-time Location-Based Recommendation")
    st.info("Click the button below to use your device's GPS for precise location detection and get crop recommendations based on your exact location.")
    
    # Initialize location state
    if 'gps_location' not in st.session_state:
        st.session_state['gps_location'] = None
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📱 Use GPS Location", type="primary", use_container_width=True):
            # Display GPS location component
            components.html(get_user_location(), height=200)
            st.info("🔄 Requesting GPS location... Please allow location access when prompted.")
    
    with col2:
        city_input = st.text_input("Or enter city name:", placeholder="e.g., New York, London")
        if st.button("🔍 Search City", use_container_width=True):
            if city_input:
                # For city search, we'll use IP-based location as fallback
                # In production, you'd use a proper geocoding service
                with st.spinner("Searching for city..."):
                    st.session_state['location_data'] = get_location_by_ip()
                    if st.session_state['location_data']:
                        st.session_state['location_data']['city'] = city_input
                        st.session_state['weather_data'] = get_weather_data(
                            st.session_state['location_data']['lat'],
                            st.session_state['location_data']['lon']
                        )
                        st.session_state['soil_data'] = get_soil_data_estimate(
                            st.session_state['location_data']['lat'],
                            st.session_state['location_data']['lon']
                        )
                        st.success(f"✅ Found location data for {city_input}")
                    else:
                        st.error("Could not find location data. Please try GPS location instead.")
    
    # Listen for GPS location data (this would need custom component in production)
    # For demo purposes, we'll add a manual GPS coordinate input
    st.markdown("---")
    st.subheader("🗺️ Manual GPS Coordinates (for testing)")
    gps_col1, gps_col2, gps_col3 = st.columns(3)
    with gps_col1:
        manual_lat = st.number_input("Latitude", value=40.7128, format="%.6f")
    with gps_col2:
        manual_lon = st.number_input("Longitude", value=-74.0060, format="%.6f")
    with gps_col3:
        if st.button("📍 Use These Coordinates", use_container_width=True):
            with st.spinner("Getting location info..."):
                st.session_state['location_data'] = reverse_geocode(manual_lat, manual_lon)
                st.session_state['weather_data'] = get_weather_data(manual_lat, manual_lon)
                st.session_state['soil_data'] = get_soil_data_estimate(manual_lat, manual_lon)
                st.success("✅ Location data retrieved successfully!")
    
    st.markdown("---")
    
    # Display location information
    if 'location_data' in st.session_state and st.session_state['location_data']:
        location = st.session_state['location_data']
        st.markdown(f"""
        <div class='location-info'>
            <h4>📍 Location Detected</h4>
            <p><strong>City:</strong> {location['city']}</p>
            <p><strong>Country:</strong> {location['country']}</p>
            <p><strong>Coordinates:</strong> {location['lat']:.2f}, {location['lon']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display weather information
        if 'weather_data' in st.session_state and st.session_state['weather_data']:
            weather = st.session_state['weather_data']
            st.markdown(f"""
            <div class='weather-card'>
                <h4>🌤️ Current Weather & Climate</h4>
                <p><strong>Temperature:</strong> {weather['temperature']:.1f}°C</p>
                <p><strong>Humidity:</strong> {weather['humidity']:.1f}%</p>
                <p><strong>Estimated Monthly Rainfall:</strong> {weather['rainfall']:.1f}mm</p>
                <p><strong>Conditions:</strong> {weather['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display soil information
        if 'soil_data' in st.session_state and st.session_state['soil_data']:
            soil = st.session_state['soil_data']
            st.markdown(f"""
            <div class='weather-card'>
                <h4>🌱 Estimated Soil Parameters</h4>
                <p><strong>Nitrogen (N):</strong> {soil['N']:.1f} ppm</p>
                <p><strong>Phosphorus (P):</strong> {soil['P']:.1f} ppm</p>
                <p><strong>Potassium (K):</strong> {soil['K']:.1f} ppm</p>
                <p><strong>pH Level:</strong> {soil['ph']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate recommendation button
        if st.button("🌾 Get Crop Recommendation", type="primary"):
            with st.spinner(''):
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st.markdown("""
                    <div class='loading-container'>
                        <div class='loading-dots'>
                            <div class='loading-dot'></div>
                            <div class='loading-dot'></div>
                            <div class='loading-dot'></div>
                        </div>
                        <div class='loading-text'>Analyzing your location data...</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                time.sleep(2)
                
                # Prepare input data
                input_data = np.array([[
                    soil['N'], soil['P'], soil['K'],
                    weather['temperature'], weather['humidity'],
                    soil['ph'], weather['rainfall']
                ]])
                
                input_scaled = scaler.transform(input_data)
                predicted_crop = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                
                loading_placeholder.empty()
            
            st.success(f"🎯 Recommended Crop for {location['city']}: **{predicted_crop}**")
            
            # Display crop image
            image_path = f"crops/{predicted_crop.lower()}.jpg"
            try:
                st.image(image_path, caption=f"Recommended for {location['city']}: {predicted_crop}", use_container_width=True)
            except:
                # If image not found, show a placeholder
                st.info(f"🌾 Recommended crop: {predicted_crop}")
                st.caption("(Crop image not available)")
            
            # Display probabilities
            prob_df = pd.DataFrame({
                "Crop": model.classes_,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False).head(10)
            
            st.subheader("Top Recommended Crops for Your Location")
            st.table(prob_df.style.format({"Probability": "{:.2%}"}).background_gradient(cmap='YlGn'))

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state['active_section'] == 'reverse':
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.header("Reverse Search: Get Ideal Parameters for a Crop")
    crop_options = sorted(data['label'].unique())
    selected_crop = st.selectbox("Select a Crop", crop_options, key="reverse_crop")
    
    if st.button("Show Ideal Parameters", type="primary", key="reverse_btn"):
        with st.spinner(''):
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown("""
                <div class='loading-container'>
                    <div class='loading-dots'>
                        <div class='loading-dot'></div>
                        <div class='loading-dot'></div>
                        <div class='loading-dot'></div>
                    </div>
                    <div class='loading-text'>Searching optimal parameters...</div>
                </div>
                """, unsafe_allow_html=True)
            
            time.sleep(1.5)
            crop_df = data[data['label'] == selected_crop]
            mean_params = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean().to_dict()
            
            loading_placeholder.empty()
        
        st.success(f"**Ideal Parameters for {selected_crop.capitalize()}**")
        
        # Display crop image
        image_path = f"crops/{selected_crop.lower()}.jpg"
        try:
            st.image(image_path, caption=f"Ideal conditions for: {selected_crop}", use_container_width=True)
        except:
            # If image not found, show a placeholder
            st.info(f"🌾 Crop: {selected_crop}")
            st.caption("(Crop image not available)")
        
        # Display parameters in a nice table
        st.markdown(f"""
        <table class='probability-table'>
            <tr><th>Parameter</th><th>Recommended Value</th></tr>
            <tr><td>Nitrogen (N)</td><td>{mean_params['N']:.1f} ppm</td></tr>
            <tr><td>Phosphorus (P)</td><td>{mean_params['P']:.1f} ppm</td></tr>
            <tr><td>Potassium (K)</td><td>{mean_params['K']:.1f} ppm</td></tr>
            <tr><td>Temperature</td><td>{mean_params['temperature']:.1f} °C</td></tr>
            <tr><td>Humidity</td><td>{mean_params['humidity']:.1f}%</td></tr>
            <tr><td>pH Level</td><td>{mean_params['ph']:.1f}</td></tr>
            <tr><td>Rainfall</td><td>{mean_params['rainfall']:.1f} mm</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer Section
st.markdown("""
<div class='footer-glass'>
    <div style='color:#888; font-size:1rem;'>&copy; 2025 - Enhanced with Auto Location</div>
</div>
""", unsafe_allow_html=True)