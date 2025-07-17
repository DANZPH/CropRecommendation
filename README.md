# 🌱 Crop Recommendation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

## [📊 Kaggle Dataset](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset "Crop Recommendation Dataset")

A modern, interactive web application that helps farmers and agricultural professionals determine the best crop to plant based on soil conditions and climate parameters. Built with advanced machine learning models and featuring a beautiful, responsive user interface.

## ✨ Features

### 🌾 **Crop Recommendation**
- **Interactive Parameter Input**: Intuitive sliders for soil nutrients (N, P, K) and climate conditions
- **Multiple ML Models**: Choose from 5 different machine learning models:
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Random Forest Classifier
  - Voting Classifier (Soft)
  - Stacking Classifier
- **Real-time Predictions**: Get instant crop recommendations with confidence levels
- **Top 5 Recommendations**: View ranked recommendations with color-coded suitability indicators
- **Visual Analytics**: Interactive charts showing crop suitability percentages

### 🔍 **Reverse Search**
- **Crop-to-Parameters Lookup**: Select any crop to see its optimal growing conditions
- **Statistical Analysis**: View mean, minimum, and maximum values for all parameters
- **Visual Crop Gallery**: High-quality crop images for better identification

### 📊 **Model Performance Analysis**
- **Comprehensive Metrics**: Accuracy, Precision, Recall, and F1-Score for all models
- **Model Comparison**: Side-by-side performance comparison
- **Detailed Descriptions**: Learn about each model's strengths and use cases

### 🎨 **Enhanced User Experience**
- **Modern Glass-morphism Design**: Beautiful, responsive interface
- **Color-coded Indicators**: 
  - 🟢 High suitability (70%+)
  - 🟡 Medium suitability (40-69%)
  - 🔴 Low suitability (<40%)
- **Loading Animations**: Smooth user experience with animated feedback
- **Mobile Responsive**: Works perfectly on all device sizes

## 🛠️ Requirements

- **Python 3.8+**
- **Core Dependencies**:
  - pandas
  - numpy
  - scikit-learn
  - streamlit
  - matplotlib
  - seaborn
  - streamlit-echarts  
  
## 🚀 Installation & Usage

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/DANZPH/CropRecommendation
cd CropRecommendation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** - The app will automatically open at `http://localhost:8501`

### 📱 How to Use

1. **Choose Your Model**: Select from 5 different ML models in the dropdown
2. **Set Parameters**: Use the intuitive sliders to input:
   - **Soil Nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)
   - **Climate Conditions**: Temperature, Humidity, pH, Rainfall
3. **Get Recommendations**: Click "Get Recommendation" for instant results
4. **Explore Features**:
   - 🌾 **Crop Recommendation**: Main prediction interface
   - 🔍 **Reverse Search**: Find optimal conditions for specific crops
   - 📊 **Model Performance**: Compare different ML models  
  
## 📊 Dataset Information

The application uses the **Crop Recommendation Dataset** containing 2,200+ samples of various crops and their optimal growing conditions:

| Parameter | Description | Unit |
|-----------|-------------|------|
| **Nitrogen (N)** | Nitrogen content in soil | ppm |
| **Phosphorus (P)** | Phosphorus content in soil | ppm |
| **Potassium (K)** | Potassium content in soil | ppm |
| **Temperature** | Average temperature | °C |
| **Humidity** | Relative humidity | % |
| **pH** | Soil acidity/alkalinity | pH scale |
| **Rainfall** | Annual rainfall | mm |
| **Label** | Crop type (22 different crops) | - |

### 🌾 Supported Crops
The system can recommend from 22 different crops including:
- **Cereals**: Rice, Maize, Wheat
- **Legumes**: Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil
- **Fruits**: Apple, Banana, Grapes, Mango, Orange, Papaya, Pomegranate, Watermelon, Muskmelon
- **Cash Crops**: Cotton, Jute, Coffee, Coconut

## 🤖 Machine Learning Models

The application features **5 advanced ML models** with ensemble learning capabilities:

### Individual Models
- **🔵 K-Nearest Neighbors (KNN)**: Pattern recognition based on similar historical cases
- **🟢 Decision Tree**: Interpretable rule-based predictions with clear decision logic
- **🟠 Random Forest**: Ensemble of decision trees with improved accuracy and robustness

### Ensemble Models
- **🟣 Voting Classifier (Soft)**: Combines KNN, Decision Tree, and Random Forest using probability averaging
- **🔴 Stacking Classifier**: Uses KNN and Decision Tree as base models with Logistic Regression meta-learner

### Model Performance
All models are trained and evaluated with comprehensive metrics including accuracy, precision, recall, and F1-score. The ensemble models typically achieve **95%+ accuracy** on the test dataset.

## 🎯 Project Structure

```
CropRecommendation/
├── app.py                 # Main Streamlit application
├── model_comparison.py    # Model evaluation and comparison
├── model.ipynb           # Jupyter notebook for analysis
├── Crop_recommendation.csv # Dataset
├── requirements.txt      # Python dependencies
├── crops/               # Crop images directory
│   ├── apple.webp
│   ├── banana.webp
│   └── ...
└── README.md           # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset)
- Built with [Streamlit](https://streamlit.io/) for the web interface
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)

---

**Made with ❤️ for sustainable agriculture and smart farming**
