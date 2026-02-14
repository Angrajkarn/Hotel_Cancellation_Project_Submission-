import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Page Config
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model not found. Please run the training script first.")
        return None

model = load_model()

# Header
st.title("🏨 Hotel Reservation Cancellation Predictor")
st.markdown("Enter reservation details to predict the likelihood of cancellation.")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=50)
        avg_price = st.number_input("Avg Price per Room ($)", min_value=0.0, max_value=1000.0, value=100.0)
        no_of_adults = st.slider("Number of Adults", 1, 5, 2)
        no_of_children = st.slider("Number of Children", 0, 5, 0)
        no_of_week_nights = st.number_input("Week Nights", 0, 20, 2)
        no_of_weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)

    with col2:
        market_segment = st.selectbox("Market Segment", 
                                      ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
        meal_plan = st.selectbox("Meal Plan", 
                                 ['Meal Plan 1', 'Not Selected', 'Meal Plan 2', 'Meal Plan 3'])
        room_type = st.selectbox("Room Type",
                                 ['Room_Type 1', 'Room_Type 4', 'Room_Type 2', 'Room_Type 6', 'Room_Type 5', 'Room_Type 7', 'Room_Type 3'])
        parking = st.selectbox("Radio Car Parking?", ["No", "Yes"])
        special_requests = st.slider("Special Requests", 0, 5, 0)
        repeated_guest = st.checkbox("Repeated Guest?")
        prev_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)

    # Derived Features hidden calculation
    total_nights = no_of_week_nights + no_of_weekend_nights
    total_guests = no_of_adults + no_of_children
    
    submitted = st.form_submit_button("Predict Cancellation Risk")

if submitted and model:
    # Prepare Input DataFrame
    input_data = pd.DataFrame({
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'type_of_meal_plan': [meal_plan],
        'required_car_parking_space': [1 if parking == "Yes" else 0],
        'room_type_reserved': [room_type],
        'lead_time': [lead_time],
        'arrival_month': [1], # Defaulting as we don't ask for date
        'market_segment_type': [market_segment],
        'repeated_guest': [1 if repeated_guest else 0],
        'no_of_previous_cancellations': [prev_cancellations],
        'no_of_previous_bookings_not_canceled': [0], # Simplified
        'avg_price_per_room': [avg_price],
        'no_of_special_requests': [special_requests],
        'total_nights': [total_nights],
        'total_guests': [total_guests]
    })
    
    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    
    if prediction == 1:
        st.error(f"🔴 High Risk of Cancellation (Probability: {probability:.1%})")
        st.markdown("**Recommendation:** Consider overbooking or contacting the guest to confirm.")
    else:
        st.success(f"🟢 Low Risk of Cancellation (Probability: {probability:.1%})")
        st.markdown("**Recommendation:** Standard reservation procedure.")
        
    # Financial Impact (Micro)
    st.markdown("### 💰 Potential Revenue Impact")
    if prediction == 1:
        loss = avg_price * total_nights
        st.write(f"Potential Revenue at Risk: **${loss:.2f}**")
    else:
        gain = avg_price * total_nights
        st.write(f"Projected Revenue: **${gain:.2f}**")
