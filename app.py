import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from streamlit_echarts import st_echarts
import time
import json

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
# In your CSS section, replace the image styling with this:
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
col_nav1, col_nav2 = st.columns(2, gap="small")
with col_nav1:
    if st.button('🌱 Crop Recommendation', key='nav_recommend', use_container_width=True):
        st.session_state['active_section'] = 'recommend'
with col_nav2:
    if st.button('🔍 best soil & climate for crops', key='nav_reverse', use_container_width=True):
        st.session_state['active_section'] = 'reverse'
st.markdown("</div>", unsafe_allow_html=True)

# Data and Model Loading
@st.cache_data
def load_data():
    data = pd.read_csv("Crop_recommendation.csv")
    return data

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
        image_path = f"crops/{predicted_crop.lower()}.webp"
        try:
            st.image(image_path, caption=predicted_crop, use_container_width=True)
        except:
            st.warning(f"Image for {predicted_crop} not found.")
        
        # Display probabilities table
        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False).head(10)  # Show top 10 only
        
        st.subheader("Top Recommended Crops")
        st.table(prob_df.style.format({"Probability": "{:.2%}"}).background_gradient(cmap='YlGn'))
        
        # Prepare chart data with rotated labels
        top_crops = prob_df.head(10)  # Show top 10 for better readability
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

if st.session_state['active_section'] == 'reverse':
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.header("Reverse Search: Get Ideal Parameters for a Crop")
    crop_options = sorted(data['label'].unique())
    selected_crop = st.selectbox("Select a Crop", crop_options, key="reverse_crop")
    
    if st.button("Show Ideal Parameters", type="primary", key="reverse_btn"):
        with st.spinner(''):
            # Custom loading animation for reverse search
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
            
            time.sleep(1.5)  # Simulate processing time
            crop_df = data[data['label'] == selected_crop]
            mean_params = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean().to_dict()
            
            loading_placeholder.empty()
        
        st.success(f"**Ideal Parameters for {selected_crop.capitalize()}**")
        
        # Display crop image
        image_path = f"crops/{selected_crop.lower()}.webp"
        try:
            st.image(image_path, caption=selected_crop, use_container_width=True)
        except:
            st.warning(f"Image for {selected_crop} not found.")
        
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
    <div style='color:#888; font-size:1rem;'>&copy; 2025 Kent Dancel</div>
</div>
""", unsafe_allow_html=True)
