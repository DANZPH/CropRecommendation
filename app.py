import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from streamlit_echarts import st_echarts
import time

st.set_page_config(page_title="Crop Prediction Model", page_icon="🌱", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) fixed;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.header-modern {
    width: 100%;
    padding: 25px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.header-modern::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><defs><radialGradient id="a" cx="50%" cy="40%"><stop offset="0%" stop-color="%23ffffff" stop-opacity="0.1"/><stop offset="100%" stop-color="%23ffffff" stop-opacity="0"/></radialGradient></defs><rect width="100" height="20" fill="url(%23a)"/></svg>');
    opacity: 0.3;
}

.header-modern h1 {
    color: #ffffff;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: 2px;
    text-align: center;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    position: relative;
    z-index: 1;
}

.header-modern .subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.1rem;
    font-weight: 400;
    margin-top: 8px;
    text-align: center;
    position: relative;
    z-index: 1;
}

.glass-card {
    background: rgba(255,255,255,0.35);
    border-radius: 18px;
    box-shadow: 0 2px 12px 0 rgba(31, 38, 135, 0.08);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 24px;
    margin: 0 auto 20px auto;
    border: 1px solid rgba(255,255,255,0.14);
}

.parameter-section {
    background: rgba(255,255,255,0.2);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    border: 1px solid rgba(255,255,255,0.3);
}

.parameter-group {
    margin-bottom: 20px;
}

.parameter-group h4 {
    color: #2c3e50;
    margin-bottom: 15px;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    background: rgba(39,174,96,0.1);
    padding: 8px;
    border-radius: 8px;
}

.stButton>button {
    border-radius: 12px;
    background-color: #27ae60;
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
    padding: 12px 30px;
    margin: 10px auto;
    display: block;
    font-size: 1.1rem;
}
.stButton>button:hover {
    background-color: #2ecc71;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(39,174,96,0.2);
}

.model-selector {
    background: rgba(255,255,255,0.3);
    border-radius: 12px;
    padding: 15px;
    margin: 15px 0;
    border: 1px solid rgba(255,255,255,0.4);
}

.confidence-badge {
    display: inline-block;
    padding: 6px 15px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-left: 10px;
}
.high-confidence {
    background-color: #27ae60;
    color: white;
}
.medium-confidence {
    background-color: #f39c12;
    color: white;
}
.low-confidence {
    background-color: #e74c3c;
    color: white;
}

.metrics-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    background: rgba(255,255,255,0.4);
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.95rem;
}
.metrics-table th, .metrics-table td {
    border: 1px solid rgba(255,255,255,0.3);
    padding: 12px 8px;
    text-align: center;
}
.metrics-table th {
    background-color: #27ae60;
    color: white;
    font-weight: 600;
}
.metrics-table tr:nth-child(even) {
    background-color: rgba(255,255,255,0.1);
}

/* Responsive crop image container */
.crop-image-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 25px auto;
    padding: 20px;
    background: rgba(255,255,255,0.25);
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    width: 100%;
    max-width: 600px;
}

/* Target Streamlit's image container specifically */
.crop-image-container > div {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

/* Desktop and larger screens */
.crop-image-container img {
    width: 100% !important;
    max-width: 450px !important;
    height: auto !important;
    min-height: 300px !important;
    max-height: 350px !important;
    border: 4px solid #27ae60 !important;
    border-radius: 15px !important;
    box-shadow: 0 12px 35px rgba(39,174,96,0.4) !important;
    object-fit: cover !important;
    display: block !important;
    margin: 0 auto !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.crop-image-container img:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 15px 45px rgba(39,174,96,0.5) !important;
}

/* Additional Streamlit image targeting */
.crop-image-container [data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

.crop-image-container [data-testid="stImage"] > img {
    width: 100% !important;
    max-width: 450px !important;
    height: auto !important;
    min-height: 300px !important;
    max-height: 350px !important;
    border: 4px solid #27ae60 !important;
    border-radius: 15px !important;
    box-shadow: 0 12px 35px rgba(39,174,96,0.4) !important;
    object-fit: cover !important;
    margin: 0 auto !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.crop-image-container [data-testid="stImage"] > img:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 15px 45px rgba(39,174,96,0.5) !important;
}

/* Force center all image elements */
.crop-image-container div {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
}

/* Target Streamlit's figure element */
.crop-image-container figure {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    margin: 0 auto !important;
    width: 100% !important;
}

/* Caption centering */
.crop-image-container figcaption {
    text-align: center !important;
    justify-content: center !important;
    display: flex !important;
    font-weight: 600 !important;
    color: #2c3e50 !important;
    margin-top: 15px !important;
    font-size: 1.1rem !important;
}

.sticky-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    color: white;
    text-align: center;
    padding: 12px 20px;
    font-size: 0.85rem;
    box-shadow: 0 -4px 15px rgba(0,0,0,0.2);
    z-index: 1000;
    backdrop-filter: blur(10px);
}

.sticky-footer a {
    color: #3498db;
    text-decoration: none;
    font-weight: 500;
}

.sticky-footer a:hover {
    color: #5dade2;
    text-decoration: underline;
}

/* Add bottom padding to body to prevent content overlap with sticky footer */
.main .block-container {
    padding-bottom: 60px;
}

.nav-buttons-container {
    margin: 0 auto 20px auto;
    max-width: 800px;
    padding: 0 20px;
}

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
    padding: 30px;
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
    font-size: 1.1rem;
}

@media (max-width: 768px) {
    .header-glass h1 {
        font-size: 1.8rem;
        padding: 0 15px;
    }
    .glass-card {
        margin: 0 10px 20px 10px;
        padding: 20px 15px;
    }
    .parameter-section {
        padding: 15px;
    }
    
    /* Mobile responsive crop image */
    .crop-image-container {
        max-width: 350px;
        padding: 15px;
        margin: 20px auto;
    }
    
    .crop-image-container img {
        max-width: 280px !important;
        min-height: 200px !important;
        max-height: 250px !important;
        border: 3px solid #27ae60 !important;
        border-radius: 12px !important;
    }
    
    .crop-image-container [data-testid="stImage"] > img {
        max-width: 280px !important;
        min-height: 200px !important;
        max-height: 250px !important;
        border: 3px solid #27ae60 !important;
        border-radius: 12px !important;
    }
    
    .crop-image-container figcaption {
        font-size: 1rem !important;
        margin-top: 10px !important;
    }
}

/* Tablet responsive */
@media (min-width: 769px) and (max-width: 1024px) {
    .crop-image-container {
        max-width: 500px;
    }
    
    .crop-image-container img {
        max-width: 380px !important;
        min-height: 250px !important;
        max-height: 300px !important;
    }
    
    .crop-image-container [data-testid="stImage"] > img {
        max-width: 380px !important;
        min-height: 250px !important;
        max-height: 300px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class='header-modern'>
    <h1>Crop Prediction Model</h1>
</div>
""", unsafe_allow_html=True)

# Navigation buttons
if 'active_section' not in st.session_state:
    st.session_state['active_section'] = 'recommend'

st.markdown("<div class='nav-buttons-container'>", unsafe_allow_html=True)
col_nav1, col_nav2, col_nav3 = st.columns(3, gap="medium")
with col_nav1:
    if st.button('🌱 Crop Recommendation', key='nav_recommend', use_container_width=True):
        st.session_state['active_section'] = 'recommend'
with col_nav2:
    if st.button('🔍 Best Soil & climate for Crops ', key='nav_reverse', use_container_width=True):
        st.session_state['active_section'] = 'reverse'
with col_nav3:
    if st.button('📊 Model Performance', key='nav_info', use_container_width=True):
        st.session_state['active_section'] = 'info'
st.markdown("</div>", unsafe_allow_html=True)

# Data and Model Loading
@st.cache_data
def load_data():
    data = pd.read_csv("Crop_recommendation.csv")
    return data

@st.cache_resource
def train_ensemble_models(data):
    """Train and return ensemble models with performance metrics"""
    # Prepare data
    le = LabelEncoder()
    data['label_encoded'] = le.fit_transform(data['label'])
    
    X = data.drop(['label', 'label_encoded'], axis=1)
    y = data['label_encoded']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Individual models
    knn_model = KNeighborsClassifier(n_neighbors=5)
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    
    # Ensemble models
    voting_soft = VotingClassifier(
        estimators=[
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
        ],
        voting='soft'
    )
    
    stacking_model = StackingClassifier(
        estimators=[
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5))
        ],
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    
    # Train all models and calculate metrics
    models = {
        'KNN': knn_model,
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
        'Voting (Soft)': voting_soft,
        'Stacking': stacking_model
    }
    
    model_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        model_metrics[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }
    
    return models, scaler, le, X_test, y_test, model_metrics

# Load data and train models
data = load_data()
models, scaler, label_encoder, X_test, y_test, model_metrics = train_ensemble_models(data)

def get_confidence_level(max_prob):
    """Determine confidence level based on maximum probability"""
    if max_prob >= 0.7:
        return "High", "high-confidence"
    elif max_prob >= 0.4:
        return "Medium", "medium-confidence"
    else:
        return "Low", "low-confidence"

def predict_with_model(input_data, model_name):
    """Make prediction with specified model"""
    model = models[model_name]
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    crop_name = label_encoder.inverse_transform([prediction])[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_scaled)[0]
        max_prob = np.max(probabilities)
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_crops = label_encoder.inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        return crop_name, max_prob, top_crops, top_probs
    else:
        return crop_name, 1.0, [crop_name], [1.0]

# Main Application Logic
if st.session_state['active_section'] == 'recommend':
    # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    # Model selection
    # st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
    model_choice = st.selectbox(
        "🤖 Choose Prediction Model:",
        options=['Voting (Soft)', 'Stacking', 'Random Forest', 'Decision Tree', 'KNN'],
        help="Ensemble models (Voting & Stacking) typically provide better accuracy"
    )
    
    # Show selected model metrics
    if model_choice in model_metrics:
        metrics = model_metrics[model_choice]
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        with col_m2:
            st.metric("Precision", f"{metrics['Precision']:.3f}")
        with col_m3:
            st.metric("Recall", f"{metrics['Recall']:.3f}")
        with col_m4:
            st.metric("F1-Score", f"{metrics['F1-score']:.3f}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input parameters with improved layout
    # st.markdown("<div class='parameter-section'>", unsafe_allow_html=True)
    st.markdown("### 🌾 Enter Soil and Climate Parameters")
    
    # Soil Nutrients Group
    st.markdown("<div class='parameter-group'>", unsafe_allow_html=True)
    st.markdown("<h4>🧪 Soil Nutrients (NPK)</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.slider("Nitrogen (N) ppm", 0, 140, 50, help="Essential for leaf growth")
    with col2:
        P = st.slider("Phosphorus (P) ppm", 5, 145, 50, help="Important for root development")
    with col3:
        K = st.slider("Potassium (K) ppm", 5, 205, 50, help="Helps with disease resistance")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Climate Conditions Group
    st.markdown("<div class='parameter-group'>", unsafe_allow_html=True)
    st.markdown("<h4>🌡️ Climate Conditions</h4>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1, help="Average temperature")
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 50.0, 0.1, help="Relative humidity")
    with col5:
        ph = st.slider("pH Level", 3.0, 10.0, 6.5, 0.1, help="Soil acidity/alkalinity")
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 1.0, help="Annual rainfall")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Get Recommendation", type="primary"):
        with st.spinner(''):
            # Loading animation
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown(f"""
                <div class='loading-container'>
                    <div class='loading-dots'>
                        <div class='loading-dot'></div>
                        <div class='loading-dot'></div>
                        <div class='loading-dot'></div>
                    </div>
                    <div class='loading-text'>Running {model_choice} analysis...</div>
                </div>
                """, unsafe_allow_html=True)
            
            time.sleep(2)
            
            # Make prediction
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            predicted_crop, max_prob, top_crops, top_probs = predict_with_model(input_data, model_choice)
            
            loading_placeholder.empty()
        
        # Display results
        confidence_level, confidence_class = get_confidence_level(max_prob)
        
        st.markdown(f"""
        <div style='text-align: center; margin: 25px 0;'>
            <h2 style='color: #27ae60; margin-bottom: 15px; font-size: 2rem;'>🌾 Recommended Crop: {predicted_crop}</h2>
            <span class='confidence-badge {confidence_class}'>
                {confidence_level} Confidence ({max_prob:.1%})
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display crop image with better styling - CENTERED
        image_path = f"crops/{predicted_crop.lower()}.webp"
        try:
            # Use wider side columns to force center column to be narrow and centered
            col_img_left, col_img_center, col_img_right = st.columns([2, 1, 2])
            with col_img_center:
                st.image(image_path, caption=f"{predicted_crop} - Recommended by {model_choice}", use_container_width=True)
        except:
            st.warning(f"🖼️ Image for {predicted_crop} not found.")
        
        # Results in two columns
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            # Top recommendations table with color indicators
            st.subheader("🏆 Top 5 Recommendations")
            
            # Create styled dataframe with color indicators
            def style_suitability(val):
                # Extract percentage value
                pct = float(val.strip('%')) / 100
                if pct >= 0.7:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'  # Green
                elif pct >= 0.4:
                    return 'background-color: #fff3cd; color: #856404; font-weight: bold'  # Yellow
                else:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'  # Red
            
            def style_confidence(val):
                if val == 'High':
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'  # Green
                elif val == 'Medium':
                    return 'background-color: #fff3cd; color: #856404; font-weight: bold'  # Yellow
                else:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'  # Red
            
            prob_df = pd.DataFrame({
                "Rank": range(1, 6),
                "Crop": top_crops,
                "Suitability": [f"{p:.1%}" for p in top_probs],
                "Confidence": [get_confidence_level(p)[0] for p in top_probs]
            })
            
            # Apply styling
            styled_df = prob_df.style.applymap(style_suitability, subset=['Suitability']) \
                                   .applymap(style_confidence, subset=['Confidence'])
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col_right:
            # Chart visualization with color indicators
            def get_bar_color(prob):
                if prob >= 70:  # High suitability (70%+)
                    return {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [{"offset": 0, "color": "#27ae60"}, {"offset": 1, "color": "#2ecc71"}]}
                elif prob >= 40:  # Medium suitability (40-69%)
                    return {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [{"offset": 0, "color": "#f39c12"}, {"offset": 1, "color": "#f1c40f"}]}
                else:  # Low suitability (<40%)
                    return {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [{"offset": 0, "color": "#e74c3c"}, {"offset": 1, "color": "#ec7063"}]}
            
            # Create data with individual colors
            chart_data = []
            for i, prob in enumerate(top_probs):
                prob_percent = round(prob * 100, 1)
                chart_data.append({
                    "value": prob_percent,
                    "itemStyle": {
                        "color": get_bar_color(prob_percent),
                        "borderRadius": [4, 4, 0, 0]
                    }
                })
            
            option = {
                "title": {"text": f"Crop Suitability - {model_choice}", "left": "center"},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "grid": {"bottom": "25%", "containLabel": True},
                "xAxis": {
                    "type": "category",
                    "data": list(top_crops),
                    "axisLabel": {"rotate": 45, "interval": 0}
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
                    "data": chart_data,
                    "label": {"show": True, "position": "top", "formatter": "{c}%"}
                }]
            }
            
            st_echarts(options=option, height="350px", key="enhanced_crop_chart")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state['active_section'] == 'reverse':
    # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    st.header("🔍 Reverse Search: Ideal Parameters")
    crop_options = sorted(data['label'].unique())
    selected_crop = st.selectbox("Select a Crop", crop_options, key="reverse_crop")
    
    if st.button("🔎 Show Ideal Parameters", type="primary", key="reverse_btn"):
        with st.spinner('Analyzing optimal conditions...'):
            time.sleep(1.5)
            
            crop_df = data[data['label'] == selected_crop]
            stats = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].agg(['mean', 'min', 'max']).round(1)
            
        st.success(f"**🌱 Optimal Growing Conditions for {selected_crop.capitalize()}**")
        
        # Display crop image
        image_path = f"crops/{selected_crop.lower()}.webp"
        try:
            # st.markdown("<div class='crop-image-container'>", unsafe_allow_html=True)
            st.image(image_path, caption=selected_crop, width=300)
            st.markdown("</div>", unsafe_allow_html=True)
        except:
            st.warning(f"🖼️ Image for {selected_crop} not found.")
        
        # Enhanced parameters table
        st.markdown(f"""
        <table class='metrics-table'>
            <tr><th>Parameter</th><th>Optimal Value</th><th>Range (Min-Max)</th></tr>
            <tr><td><strong>Nitrogen (N)</strong></td><td>{stats.loc['mean', 'N']} ppm</td><td>{stats.loc['min', 'N']} - {stats.loc['max', 'N']} ppm</td></tr>
            <tr><td><strong>Phosphorus (P)</strong></td><td>{stats.loc['mean', 'P']} ppm</td><td>{stats.loc['min', 'P']} - {stats.loc['max', 'P']} ppm</td></tr>
            <tr><td><strong>Potassium (K)</strong></td><td>{stats.loc['mean', 'K']} ppm</td><td>{stats.loc['min', 'K']} - {stats.loc['max', 'K']} ppm</td></tr>
            <tr><td><strong>Temperature</strong></td><td>{stats.loc['mean', 'temperature']} °C</td><td>{stats.loc['min', 'temperature']} - {stats.loc['max', 'temperature']} °C</td></tr>
            <tr><td><strong>Humidity</strong></td><td>{stats.loc['mean', 'humidity']}%</td><td>{stats.loc['min', 'humidity']} - {stats.loc['max', 'humidity']}%</td></tr>
            <tr><td><strong>pH Level</strong></td><td>{stats.loc['mean', 'ph']}</td><td>{stats.loc['min', 'ph']} - {stats.loc['max', 'ph']}</td></tr>
            <tr><td><strong>Rainfall</strong></td><td>{stats.loc['mean', 'rainfall']} mm</td><td>{stats.loc['min', 'rainfall']} - {stats.loc['max', 'rainfall']} mm</td></tr>
        </table>
        """, unsafe_allow_html=True)
        
        st.info(f"📊 Based on {len(crop_df)} samples from the dataset")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state['active_section'] == 'info':
    # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    st.header("📊 Model Performance Analysis")
    
    # Model Performance Metrics Table
    st.subheader("🎯 Model Performance Comparison")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df = metrics_df.round(4)
    
    # Display as formatted table
    st.markdown(f"""
    <table class='metrics-table'>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
        </tr>
        <tr>
            <td><strong>KNN</strong></td>
            <td>{metrics_df.loc['KNN', 'Accuracy']:.4f}</td>
            <td>{metrics_df.loc['KNN', 'Precision']:.4f}</td>
            <td>{metrics_df.loc['KNN', 'Recall']:.4f}</td>
            <td>{metrics_df.loc['KNN', 'F1-score']:.4f}</td>
        </tr>
        <tr>
            <td><strong>Decision Tree</strong></td>
            <td>{metrics_df.loc['Decision Tree', 'Accuracy']:.4f}</td>
            <td>{metrics_df.loc['Decision Tree', 'Precision']:.4f}</td>
            <td>{metrics_df.loc['Decision Tree', 'Recall']:.4f}</td>
            <td>{metrics_df.loc['Decision Tree', 'F1-score']:.4f}</td>
        </tr>
        <tr>
            <td><strong>Random Forest</strong></td>
            <td>{metrics_df.loc['Random Forest', 'Accuracy']:.4f}</td>
            <td>{metrics_df.loc['Random Forest', 'Precision']:.4f}</td>
            <td>{metrics_df.loc['Random Forest', 'Recall']:.4f}</td>
            <td>{metrics_df.loc['Random Forest', 'F1-score']:.4f}</td>
        </tr>
        <tr>
            <td><strong>Voting (Soft)</strong></td>
            <td>{metrics_df.loc['Voting (Soft)', 'Accuracy']:.4f}</td>
            <td>{metrics_df.loc['Voting (Soft)', 'Precision']:.4f}</td>
            <td>{metrics_df.loc['Voting (Soft)', 'Recall']:.4f}</td>
            <td>{metrics_df.loc['Voting (Soft)', 'F1-score']:.4f}</td>
        </tr>
        <tr>
            <td><strong>Stacking</strong></td>
            <td>{metrics_df.loc['Stacking', 'Accuracy']:.4f}</td>
            <td>{metrics_df.loc['Stacking', 'Precision']:.4f}</td>
            <td>{metrics_df.loc['Stacking', 'Recall']:.4f}</td>
            <td>{metrics_df.loc['Stacking', 'F1-score']:.4f}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)
    
    # Find and highlight best model
    best_model = metrics_df['Accuracy'].idxmax()
    best_accuracy = metrics_df.loc[best_model, 'Accuracy']
    
    st.success(f"🏆 **Best Performing Model: {best_model}** with {best_accuracy:.4f} accuracy")
    
    # Model Information
    st.subheader("🤖 Model Descriptions")
    
    model_info = {
        "KNN": {
            "description": "K-Nearest Neighbors finds similar historical cases for prediction",
            "strength": "Simple, works well with local patterns and clusters",
            "use_case": "Good for datasets with clear regional patterns",
            "complexity": "Low"
        },
        "Decision Tree": {
            "description": "Single tree-based model with clear decision rules",
            "strength": "Highly interpretable, fast predictions, shows decision logic",
            "use_case": "When you need to understand exactly why a crop was recommended",
            "complexity": "Low"
        },
        "Random Forest": {
            "description": "Ensemble of multiple decision trees with voting",
            "strength": "Handles overfitting well, provides feature importance",
            "use_case": "Reliable baseline with good interpretability",
            "complexity": "Medium"
        },
        "Voting (Soft)": {
            "description": "Combines KNN, Decision Tree, and Random Forest using probability averaging",
            "strength": "Best overall accuracy, robust predictions, reduces individual model weaknesses",
            "use_case": "Recommended for most accurate results in production",
            "complexity": "High"
        },
        "Stacking": {
            "description": "Uses KNN and Decision Tree as base models with Logistic Regression meta-learner",
            "strength": "Learns optimal combination of base models automatically",
            "use_case": "Good for complex pattern recognition and when you want automated model combination",
            "complexity": "High"
        }
    }
    
    for model_name, info in model_info.items():
        with st.expander(f"🔧 {model_name} - Accuracy: {metrics_df.loc[model_name, 'Accuracy']:.4f}"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Strength:** {info['strength']}")
            st.write(f"**Best Use Case:** {info['use_case']}")
            st.write(f"**Complexity:** {info['complexity']}")
            
            # Show all metrics for this model
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics_df.loc[model_name, 'Accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics_df.loc[model_name, 'Precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics_df.loc[model_name, 'Recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics_df.loc[model_name, 'F1-score']:.4f}")
    
    # Dataset Information
    st.subheader("📈 Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{len(data):,}")
    with col2:
        st.metric("Number of Crops", data['label'].nunique())
    with col3:
        st.metric("Features", len(data.columns) - 1)
    
    st.write(f"**Features:** {', '.join(data.columns[:-1])}")
    
    # Crop distribution chart
    crop_counts = data['label'].value_counts().head(10)
    
    chart_data = {
        "title": {"text": "Top 10 Crops in Dataset", "left": "center"},
        "tooltip": {"trigger": "item"},
        "series": [{
            "name": "Crop Distribution",
            "type": "pie",
            "radius": "60%",
            "data": [{"value": int(count), "name": crop} for crop, count in crop_counts.items()],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                }
            }
        }]
    }
    
    st_echarts(options=chart_data, height="400px", key="crop_distribution")

    st.markdown("</div>", unsafe_allow_html=True)

# Sticky Footer
st.markdown("""
<div class='sticky-footer'>
    © 2025Crops Recommendation System
</div>
""", unsafe_allow_html=True)