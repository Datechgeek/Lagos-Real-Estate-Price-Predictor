import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Set page config
st.set_page_config(
    page_title="Lagos Real Estate Price Predictor", page_icon="🏘️", layout="wide"
)

# Define paths - use relative paths for GitHub
model_dir = Path("models")
model_path = model_dir / "xgboost_lagos_housing_model.joblib"
town_tiers_path = model_dir / "town_tiers.joblib"
data_path = Path("data") / "nigeria_houses_data.csv"

# Try to load the model, or train a new one if files aren't found
@st.cache_resource
def get_model_and_tiers():
    try:
        # Try to load saved model and tiers
        if os.path.exists(model_path) and os.path.exists(town_tiers_path):
            model = joblib.load(model_path)
            town_tiers = joblib.load(town_tiers_path)
            st.success("✅ Model loaded successfully!")
            return model, town_tiers
        else:
            raise FileNotFoundError(f"Model files not found at {model_path}")
    except Exception as e:
        st.warning(f"⚠️ Could not load model files: {e}")
        st.info("Training a new model... (this may take a minute)")
        
        # Load data for training
        try:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
            else:
                # Try alternative path formats
                alt_path = "data/nigeria_houses_data.csv"
                df = pd.read_csv(alt_path)
                
            df = df[df["state"] == "Lagos"]
        except Exception as data_e:
            st.error(f"❌ Could not load data: {data_e}")
            st.stop()
        
        # Create location tiers
        town_stats = df.groupby("town")["price"].agg(["median"]).reset_index()
        town_tiers = {}
        all_towns_median = df["price"].median()
        
        for _, row in town_stats.iterrows():
            town = row["town"]
            median = row["median"]
            
            if median > (all_towns_median * 2):
                town_tiers[town] = "Premium"
            elif median > all_towns_median:
                town_tiers[town] = "Above Average"
            elif median > (all_towns_median * 0.5):
                town_tiers[town] = "Average"
            else:
                town_tiers[town] = "Value"
        
        # Add location tier to dataframe
        df["location_tier"] = df["town"].map(town_tiers)
        
        # Split data
        X = df.drop(columns=["price"])
        y = df["price"]
        
        # Train new model
        model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            SimpleImputer(),
            XGBRegressor(random_state=42)
        )
        model.fit(X, y)
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model for future use if possible
        try:
            joblib.dump(model, model_path)
            joblib.dump(town_tiers, town_tiers_path)
            st.success("✅ Model saved for future use!")
        except Exception as save_e:
            st.warning(f"⚠️ Could not save model: {save_e}")
        
        st.success("✅ New model trained successfully!")
        return model, town_tiers

# Get model and town tiers
model, town_tiers = get_model_and_tiers()

# Load dataset to get dropdown options
def load_dataset_info():
    try:
        # Try multiple potential paths for GitHub deployment
        for potential_path in [data_path, "data/nigeria_houses_data.csv", "../data/nigeria_houses_data.csv"]:
            if os.path.exists(potential_path):
                df = pd.read_csv(potential_path)
                break
        else:
            # Fallback to using town_tiers if data file not found
            st.warning("Using fallback options - please ensure data is uploaded to your repository")
            return {
                "towns": list(town_tiers.keys()),
                "property_types": ["Flat", "House", "Apartment", "Duplex", "Bungalow"],
                "bedrooms_range": range(1, 6),
                "bathrooms_range": range(1, 6),
                "toilets_range": range(1, 6),
                "parking_range": range(0, 4),
            }
        
        # Filter for Lagos properties
        df = df[df["state"] == "Lagos"]

        # Get unique values and ranges
        towns = sorted(df["town"].dropna().unique().tolist())
        # Remove Ajah from the list of towns
        towns = [town for town in towns if town != "Ajah"]

        property_types = sorted(df["title"].dropna().unique().tolist())

        # Get numeric ranges with sensible limits
        bedrooms_range = range(1, min(int(df["bedrooms"].max()) + 1, 10))
        bathrooms_range = range(1, min(int(df["bathrooms"].max()) + 1, 10))
        toilets_range = range(1, min(int(df["toilets"].max()) + 1, 10))
        parking_range = range(0, min(int(df["parking_space"].max()) + 1, 6))

        return {
            "towns": towns,
            "property_types": property_types,
            "bedrooms_range": bedrooms_range,
            "bathrooms_range": bathrooms_range,
            "toilets_range": toilets_range,
            "parking_range": parking_range,
        }
    except Exception as e:
        st.warning(f"Using fallback values - could not load dataset: {e}")
        # Fallback values
        return {
            "towns": list(town_tiers.keys()),
            "property_types": ["Flat", "House", "Apartment", "Duplex", "Bungalow"],
            "bedrooms_range": range(1, 6),
            "bathrooms_range": range(1, 6),
            "toilets_range": range(1, 6),
            "parking_range": range(0, 4),
        }

# Function to format large numbers as naira
def format_naira(amount):
    if amount >= 1_000_000_000:
        return f"₦{amount / 1_000_000_000:.2f} Billion"
    elif amount >= 1_000_000:
        return f"₦{amount / 1_000_000:.2f} Million"
    else:
        return f"₦{amount:,.2f}"

# Get dataset information
dataset_info = load_dataset_info()

# Title and introduction with emojis
st.title("🏘️ Lagos Real Estate Price Predictor 🏢")
st.markdown("### 🔍 Find the value of your property in Lagos, Nigeria 💰")

# Create columns for inputs
col1, col2 = st.columns(2)

# First column - Location details
with col1:
    st.markdown("### 📍 Location Details")

    # Town selection
    selected_town = st.selectbox("🏙️ Select Town/Area", dataset_info["towns"])

    # Get location tier based on selected town
    location_tier = town_tiers.get(selected_town, "Average")

    # Display the location tier with emoji
    tier_emojis = {
        "Premium": "💎",
        "Above Average": "⭐",
        "Average": "✅",
        "Value": "💲",
    }
    tier_emoji = tier_emojis.get(location_tier, "🏠")

    st.info(f"{tier_emoji} Location Tier: {location_tier}")

    # Property type selection
    selected_property_type = st.selectbox(
        "🏗️ Property Type", dataset_info["property_types"]
    )

# Second column - Property features
with col2:
    st.markdown("### 🏠 Property Features")

    # Bedrooms
    bedrooms = st.selectbox(
        "🛏️ Number of Bedrooms", list(dataset_info["bedrooms_range"])
    )

    # Bathrooms
    bathrooms = st.selectbox(
        "🚿 Number of Bathrooms", list(dataset_info["bathrooms_range"])
    )

    # Toilets
    toilets = st.selectbox("🚽 Number of Toilets", list(dataset_info["toilets_range"]))

    # Parking spaces
    parking_spaces = st.selectbox(
        "🚗 Number of Parking Spaces", list(dataset_info["parking_range"])
    )

# Create a divider
st.markdown("---")

# Prediction button
prediction_col1, prediction_col2, prediction_col3 = st.columns([1, 2, 1])
with prediction_col2:
    predict_button = st.button(
        "💰 Calculate Property Value 💰", use_container_width=True
    )

# Show prediction results
if predict_button:
    with st.spinner("🔄 Calculating property value..."):
        # Create input dataframe for prediction
        input_data = pd.DataFrame(
            {
                "bedrooms": [bedrooms],
                "bathrooms": [bathrooms],
                "toilets": [toilets],
                "parking_space": [parking_spaces],
                "title": [selected_property_type],
                "town": [selected_town],
                "state": ["Lagos"],
                "location_tier": [location_tier],
            }
        )

        try:
            # Ensure columns are in the exact same order as during training
            expected_columns = [
                "bedrooms",
                "bathrooms",
                "toilets",
                "parking_space",
                "title",
                "town",
                "state",
                "location_tier",
            ]

            # Reorder columns to match training order
            input_data = input_data[expected_columns]

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Create price range (±15%)
            lower_bound = prediction * 0.85
            upper_bound = prediction * 1.15

            # Display the results in a nice card
            st.success("### 🎯 Property Valuation Complete!")

            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

            with result_col2:
                # Show the estimated value in a highlighted box
                st.markdown(
                    """
                <div style="background-color:#E8F0FE; padding:20px; border-radius:10px; text-align:center;">
                    <h3>💰 Estimated Property Value</h3>
                    <h2 style="color:#1E88E5;">{}</h2>
                    <p>Estimated Price Range:<br>{} - {}</p>
                </div>
                """.format(
                        format_naira(prediction),
                        format_naira(lower_bound),
                        format_naira(upper_bound),
                    ),
                    unsafe_allow_html=True,
                )

                # Display property specifications
                st.markdown("### 📋 Property Details Summary")
                specs_df = pd.DataFrame(
                    {
                        "Feature": [
                            "Location 📍",
                            "Property Type 🏠",
                            "Location Tier " + tier_emoji,
                            "Bedrooms 🛏️",
                            "Bathrooms 🚿",
                            "Toilets 🚽",
                            "Parking Spaces 🚗",
                        ],
                        "Value": [
                            selected_town,
                            selected_property_type,
                            location_tier,
                            bedrooms,
                            bathrooms,
                            toilets,
                            parking_spaces,
                        ],
                    }
                )
                st.table(specs_df)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.warning("Please try different property specifications.")

# Footer section with information
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 About This Tool")
    st.markdown("""
    This real estate price predictor uses XGBoost machine learning to estimate 
    property values in Lagos. The model considers location quality, property features, 
    and amenities to determine an appropriate price range.
    
    **Note**: Predictions are estimates and actual market values may vary based on factors 
    like property condition, exact location, and current market trends.
    """)

with col2:
    st.markdown("### 🏙️ Lagos Property Market Insights")
    st.markdown("""
    **Premium Locations** 💎: Ikoyi, Victoria Island, Lekki Phase 1
    
    **Above Average** ⭐: Gbagada, Magodo, Ikeja GRA
    
    **Average Areas** ✅: Surulere, Yaba, Mainland areas
    
    **Value Areas** 💲: Outskirts with growing infrastructure
    
    Property values are significantly influenced by proximity to business districts,
    infrastructure quality, and neighborhood amenities.
    """)

# Final footer
st.markdown("---")
st.markdown("### 🏘️ Lagos Real Estate Price Predictor © 2025")
