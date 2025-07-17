# Crop Recommendation System  
  ## [Kaggle Dataset](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset "Crop Recommendation Dataset")
This application helps farmers determine the best crop to plant based on soil conditions and climate parameters. It uses a machine learning model trained on a dataset of various crops and their optimal growing conditions.  
  
## Features  
  
- Input soil parameters (N, P, K, pH)  
- Input climate parameters (temperature, humidity, rainfall)  
- Get crop recommendations based on the provided parameters  
- View top 5 recommended crops with suitability scores  
- Explore the dataset used for training  
  
## Requirements  
  
- Python 3.6+  
- pandas  
- numpy  
- scikit-learn  
- streamlit  
- matplotlib  
- seaborn  
- streamlit-echarts  
  
## Installation & Usage
  
1. Clone this repository or download the files  
```bash  
git clone https://github.com/DANZPH/CropRecommendation  
```
2. Navigate to the project directory  
```bash  
cd CropRecommendation   
```
3. Install the required packages:  
```bash  
pip install -r requirements.txt  
```
2. Run the Streamlit app:  
```bash  
streamlit run app.py  
```  
  
3. The application will open in your default web browser  
4. Enter your soil and climate parameters in the sidebar  
5. Click "Get Recommendation" to see the results  
  
## Dataset  
  
The model is trained on the Crop Recommendation dataset, which contains information about various crops and their optimal growing conditions. The dataset includes the following parameters:  
  
- Nitrogen (N): Nitrogen content in the soil (in kg/ha)  
- Phosphorus (P): Phosphorus content in the soil (in kg/ha)  
- Potassium (K): Potassium content in the soil (in kg/ha)  
- Temperature: Temperature in degrees Celsius  
- Humidity: Relative humidity in percentage  
- pH: pH value of the soil  
- Rainfall: Rainfall in millimeters  
- Label: Type of crop  
  
## Model  
  
The application uses two machine learning models for crop recommendation and analysis:  
  
- **K-Nearest Neighbors (KNN):** Used for comparison and evaluation in the notebook. KNN predicts the crop based on the majority class among the nearest data points in the feature space.  
- **Decision Tree Classifier:** Used in the main Streamlit app for crop prediction. Decision Trees split the data based on feature values to make interpretable predictions.  
  
Both models are trained and evaluated in the included notebook (`FinalProject.ipynb`). The Decision Tree model is used for real-time recommendations in the app, while KNN is included for performance comparison and educational purposes.  
  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
