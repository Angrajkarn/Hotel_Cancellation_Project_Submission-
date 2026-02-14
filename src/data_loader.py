import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_samples=10000, save_path=None):
    """
    Generates a synthetic dataset for hotel reservation cancellations.
    
    Args:
        num_samples (int): Number of rows to generate.
        save_path (str, optional): Path to save the CSV file.
        
    Returns:
        pd.DataFrame: The generated dataframe.
    """
    np.random.seed(42)
    random.seed(42)
    
    # booking_ids
    booking_ids = [f"INN{i:05d}" for i in range(1, num_samples + 1)]
    
    # no_of_adults: Mostly 2, some 1 or 3
    no_of_adults = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.25, 0.65, 0.08, 0.02])
    
    # no_of_children: Mostly 0
    no_of_children = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.9, 0.06, 0.03, 0.01])
    
    # stay duration
    no_of_weekend_nights = np.random.choice(range(0, 5), size=num_samples, p=[0.4, 0.3, 0.2, 0.08, 0.02])
    no_of_week_nights = np.random.choice(range(0, 10), size=num_samples)
    
    # meal_type
    meal_types = ['Meal Plan 1', 'Meal Plan 2', 'Not Selected', 'Meal Plan 3']
    meal_type = np.random.choice(meal_types, size=num_samples, p=[0.7, 0.2, 0.09, 0.01])
    
    # car parking
    required_car_parking = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
    
    # lead_time (0 to 300 days) - skewed distribution
    lead_time = np.random.gamma(shape=2, scale=30, size=num_samples).astype(int)
    lead_time = np.clip(lead_time, 0, 400)
    
    # arrival dates
    start_date = datetime(2023, 1, 1)
    arrival_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_samples)]
    arrival_year = [d.year for d in arrival_dates]
    arrival_month = [d.month for d in arrival_dates]
    arrival_date = [d.day for d in arrival_dates]
    
    # market_segment
    segments = ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation']
    market_segment = np.random.choice(segments, size=num_samples, p=[0.6, 0.25, 0.1, 0.01, 0.04])
    
    # repeated_guest
    repeated_guest = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
    
    # history
    no_previous_cancellations = np.zeros(num_samples, dtype=int)
    previous_bookings_not_canceled = np.zeros(num_samples, dtype=int)
    
    # Logic for repeated guests using numpy indexing
    repeated_indices = np.where(repeated_guest == 1)[0]
    num_repeated = len(repeated_indices)
    
    # Assign history only to repeated guests
    if num_repeated > 0:
        no_previous_cancellations[repeated_indices] = np.random.poisson(0.5, size=num_repeated)
        previous_bookings_not_canceled[repeated_indices] = np.random.poisson(2, size=num_repeated)

    
    # avg_price_per_room
    avg_price = np.random.normal(100, 30, size=num_samples)
    avg_price = np.clip(avg_price, 20, 500).round(2)
    
    # special requests
    no_of_special_requests = np.random.choice(range(6), size=num_samples, p=[0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
    
    # --- Target Generation (booking_status) with correlations ---
    # Base probability
    prob_cancellation = np.full(num_samples, 0.3)
    
    # Adjust probability based on features
    
    # Lead time: Higher lead time -> higher cancellation
    prob_cancellation += (lead_time / 300) * 0.4
    
    # Market segment: Online higher cancellation, Corporate lower
    prob_cancellation[market_segment == 'Online'] += 0.1
    prob_cancellation[market_segment == 'Corporate'] -= 0.1
    prob_cancellation[market_segment == 'Complementary'] -= 0.2
    
    # Special requests: More requests -> less likely to cancel
    prob_cancellation -= (no_of_special_requests * 0.05)
    
    # Price: Higher price -> slightly higher cancellation
    prob_cancellation += (avg_price / 300) * 0.1
    
    # Repeated guest: Less likely to cancel
    prob_cancellation[repeated_guest == 1] -= 0.15
    
    # Previous cancellations: More previous cancellations -> higher risk
    prob_cancellation += (no_previous_cancellations * 0.1)
    
    # Clip probabilities
    prob_cancellation = np.clip(prob_cancellation, 0.05, 0.95)
    
    # Generate target
    booking_status_labels = ['Not_Canceled', 'Canceled']
    booking_status = [np.random.choice(booking_status_labels, p=[1-p, p]) for p in prob_cancellation]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Booking_ID': booking_ids,
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'meal_type': meal_type,
        'required_car_parking_spaces': required_car_parking,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment': market_segment,
        'repeated_guest': repeated_guest,
        'no_previous_cancellations': no_previous_cancellations,
        'previous_bookings_not_canceled': previous_bookings_not_canceled,
        'avg_price_per_room': avg_price,
        'no_of_special_requests': no_of_special_requests,
        'booking_status': booking_status
    })
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
        
    return df

if __name__ == "__main__":
    generate_synthetic_data(save_path=r"c:\\Users\\Nivedita\\.gemini\\antigravity\\playground\\temporal-aurora\\hotel_cancellations\\data\\synthetic_hotel_data.csv")
