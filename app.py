import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Investment Advisor")
st.write(
    "Explore a property using the project's trained machine-learning models. "
    "The investment label and five-year price are constructed proxies/scenarios, "
    "not verified investment outcomes or market forecasts."
)

CLASSIFIER_PATH = "investment_classifier.joblib"
REGRESSOR_PATH = "price_regressor.joblib"
DATA_PATH = "housing_data_with_targets.csv"

@st.cache_resource
def load_models():
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    return classifier, regressor

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(REGRESSOR_PATH):
    st.error(
        "Model files are missing. Place investment_classifier.joblib and "
        "price_regressor.joblib in the same folder as app.py."
    )
    st.stop()

try:
    classifier, regressor = load_models()
except Exception as exc:
    st.error(f"Could not load model files: {exc}")
    st.stop()

reference_data = load_data()

def get_options(column, fallback):
    if reference_data is not None and column in reference_data.columns:
        vals = reference_data[column].dropna().astype(str).unique().tolist()
        if vals:
            return sorted(vals)
    return fallback

def get_num_range(column, default_min, default_max, default_value):
    if reference_data is not None and column in reference_data.columns:
        series = pd.to_numeric(reference_data[column], errors="coerce").dropna()
        if len(series):
            lo = float(series.quantile(0.01))
            hi = float(series.quantile(0.99))
            if lo < hi:
                return lo, hi, float(np.clip(default_value, lo, hi))
    return float(default_min), float(default_max), float(default_value)

st.sidebar.header("Property details")
with st.sidebar.form("property_form"):
    st.subheader("Location")
    state = st.selectbox("State", get_options("State", ["Other"]))
    city = st.selectbox("City", get_options("City", ["Other"]))
    locality = st.selectbox("Locality", get_options("Locality", ["Other"]))

    st.subheader("Property")
    property_type = st.selectbox(
        "Property type", get_options("Property_Type", ["Apartment", "Villa", "Independent House"])
    )
    bhk_min, bhk_max, bhk_default = get_num_range("BHK", 1, 10, 2)
    bhk = st.number_input("BHK", min_value=1, max_value=20, value=int(round(bhk_default)), step=1)

    size_min, size_max, size_default = get_num_range("Size_in_SqFt", 100, 10000, 1200)
    size_sqft = st.number_input(
        "Size (sq ft)", min_value=1, max_value=100000,
        value=int(round(size_default)), step=50
    )

    price_min, price_max, price_default = get_num_range("Price_in_Lakhs", 1, 10000, 75)
    price_lakhs = st.number_input(
        "Current price (₹ lakhs)", min_value=0.1, max_value=100000.0,
        value=float(price_default), step=1.0
    )

    year_min, year_max, year_default = get_num_range("Year_Built", 1900, 2026, 2015)
    year_built = st.number_input(
        "Year built", min_value=1800, max_value=2026,
        value=int(round(year_default)), step=1
    )

    furnishing = st.selectbox(
        "Furnishing status",
        get_options("Furnished_Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
    )
    floor_no = st.number_input("Floor number", min_value=0, max_value=200, value=1, step=1)
    total_floors = st.number_input("Total floors", min_value=1, max_value=200, value=5, step=1)
    nearby_schools = st.number_input("Nearby schools", min_value=0, max_value=100, value=3, step=1)
    nearby_hospitals = st.number_input("Nearby hospitals", min_value=0, max_value=100, value=2, step=1)

    st.subheader("Facilities and status")
    transport = st.selectbox(
        "Public transport accessibility",
        get_options("Public_Transport_Accessibility", ["Low", "Medium", "High"])
    )
    parking = st.selectbox("Parking space", get_options("Parking_Space", ["Yes", "No"]))
    security = st.selectbox("Security", get_options("Security", ["Yes", "No"]))
    amenities = st.text_input("Amenities (comma-separated)", "Gym, Garden")
    facing = st.selectbox("Facing", get_options(
        "Facing", ["North", "South", "East", "West", "North-East", "North-West", "South-East", "South-West"]
    ))
    owner_type = st.selectbox("Owner type", get_options("Owner_Type", ["Owner", "Agent", "Builder"]))
    availability = st.selectbox(
        "Availability status",
        get_options("Availability_Status", ["Ready_to_Move", "Under_Construction"])
    )

    submitted = st.form_submit_button("Analyze property", use_container_width=True)

if submitted:
    current_year = pd.Timestamp.now().year
    property_age = max(0, current_year - int(year_built))

    # These columns must match the model's training input columns.
    input_df = pd.DataFrame([{
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": int(bhk),
        "Size_in_SqFt": int(size_sqft),
        "Price_in_Lakhs": float(price_lakhs),
        "Year_Built": int(year_built),
        "Furnished_Status": furnishing,
        "Floor_No": int(floor_no),
        "Total_Floors": int(total_floors),
        "Age_of_Property": int(property_age),
        "Nearby_Schools": int(nearby_schools),
        "Nearby_Hospitals": int(nearby_hospitals),
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability
    }])

    try:
        prediction = int(classifier.predict(input_df)[0])
        predicted_future_price = float(regressor.predict(input_df)[0])

        st.header("Analysis results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Investment proxy",
                "Below locality median" if prediction == 1 else "Not below locality median"
            )
            st.caption(
                "This is the model's classification based on the constructed label; "
                "it is not a recommendation to buy."
            )

        with col2:
            st.metric(
                "Five-year scenario price",
                f"₹{predicted_future_price:,.2f} lakh"
            )
            st.caption(
                "This is based on a dataset target created using an assumed 6% annual "
                "appreciation rate; it is not a real market forecast."
            )

        with st.expander("View submitted property details"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that the input columns and their spelling match the columns used "
            "when training the models."
        )

st.divider()
st.caption(
    "Educational project prototype. Property prices and investment decisions depend "
    "on market conditions, location, legal checks, financing, and other factors."
)
