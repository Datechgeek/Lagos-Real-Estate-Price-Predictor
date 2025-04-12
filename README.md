# Lagos-Real-Estate-Price-Predictor
This project predicts property prices in Lagos, Nigeria based on features like location, property type, and amenities using machine learning. Users can estimate property values through an interactive Streamlit web application.

Features
🏠 Predict real estate prices in Lagos with machine learning
📍 Location-based pricing using premium tier categorization
🏢 Support for various property types (houses, apartments, duplexes, etc.)
🔍 Interactive inputs for bedrooms, bathrooms, and other property features
💰 Price estimates with confidence ranges
📊 Visual representation of results

How It Works
1. Data Analysis and Preparation
The system analyzes property data from Lagos, classifying locations into price tiers (Premium, Above Average, Average, Value) based on median property values in each area.

2. Machine Learning Model
We trained and compared multiple regression models including:

Linear Regression
Ridge Regression
Random Forest
LightGBM
XGBoost
XGBoost delivered the best performance and was selected as our production model.

3. Web Application
Our Streamlit app allows users to:

Select property location from Lagos towns
Choose property type
Specify number of bedrooms, bathrooms, toilets
Set number of parking spaces
Get instant price predictions with confidence intervals
Technologies Used
Python: Core programming language
Pandas/NumPy: Data manipulation
Scikit-learn: ML pipelines and preprocessing
XGBoost/LightGBM: Advanced regression models
Matplotlib/Seaborn: Data visualization
Streamlit: Web application framework
Joblib: Model serialization

Installation
Clone this repository
git clone <repository-url>

Install required packages
pip install -r requirements.txt

Run the Streamlit app
streamlit run app.py

Project Structure
Real Estate in Lagos/
├── data/
│   └── nigeria_houses_data.csv     # Property data
├── models/                         # Trained models
│   ├── xgboost_lagos_housing_model.joblib
│   └── town_tiers.joblib           # Location tier mappings
├── Notebooks/                      # Analysis notebooks
│   └── Real_Estate_in_Lagos_Final.ipynb
├── app.py                          # Streamlit web application
└── requirements.txt                # Project dependencies

Future Improvements
Add more property features (year built, renovation status)
Implement time-based price trends
Create neighborhood comparison tools
Add interactive maps for location selection
Expand to other Nigerian cities

About
This project was created to help property buyers, sellers, and real estate professionals better understand and estimate property values in Lagos, Nigeria using data-driven approaches.
