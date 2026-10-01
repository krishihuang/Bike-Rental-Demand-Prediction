import streamlit as st
import pandas as pd
import joblib
from datetime import date


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Bike Rental Predictor",
    page_icon="🚲",
    layout="centered"
)


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("bike_model.pkl")


model = load_model()


# -----------------------------
# Helper function
# -----------------------------

def get_season(month):
    if month in [1, 2, 12]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8, 9]:
        return 3
    else:
        return 4


# -----------------------------
# Page heading
# -----------------------------

st.title("🚲 Bike Rental Demand Predictor")

st.write(
    "Enter weather and calendar information to estimate "
    "the number of bike rentals."
)

st.info(
    "The model was trained on the UCI Bike Sharing dataset. "
    "Weather variables are represented using the dataset's "
    "normalized 0–1 scale."
)


# -----------------------------
# Date information
# -----------------------------

st.subheader("📅 Date")

selected_date = st.date_input(
    "Select a date",
    value=date(2012, 6, 15)
)

year_value = selected_date.year
month_value = selected_date.month

# Dataset uses:
# 0 = 2011
# 1 = 2012
yr = 0 if year_value == 2011 else 1

mnth = month_value

# Python weekday:
# Monday = 0 ... Sunday = 6
weekday = selected_date.weekday()

season = get_season(month_value)


# -----------------------------
# Calendar information
# -----------------------------

st.subheader("📆 Calendar")

holiday = st.selectbox(
    "Is this a holiday?",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

workingday = st.selectbox(
    "Is this a working day?",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)


# -----------------------------
# Weather information
# -----------------------------

st.subheader("🌤️ Weather")

temp = st.slider(
    "Temperature (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

atemp = st.slider(
    "Feels-like temperature (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

hum = st.slider(
    "Humidity (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

windspeed = st.slider(
    "Wind speed (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

weathersit = st.selectbox(
    "Weather condition",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Clear / Few clouds",
        2: "Mist / Cloudy",
        3: "Light rain / Snow",
        4: "Heavy rain / Snow"
    }[x]
)


# -----------------------------
# Prediction
# -----------------------------

st.divider()

if st.button("Predict Bike Rentals", type="primary"):

    customer = pd.DataFrame([{
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed
    }])

    prediction = model.predict(customer)[0]

    # Rental count cannot logically be negative
    prediction = max(0, prediction)

    st.subheader("📊 Prediction")

    st.metric(
        "Estimated bike rentals",
        f"{prediction:,.0f}"
    )

    st.success(
        f"The model estimates approximately "
        f"{prediction:,.0f} rentals for the selected conditions."
    )

    st.caption(
        "This is a machine-learning estimate based on historical "
        "bike-sharing data and should not be interpreted as a guarantee."
    )