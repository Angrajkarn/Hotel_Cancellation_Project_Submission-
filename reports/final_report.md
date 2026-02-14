# Hotel Reservation Cancellation Prediction - Final Report

## 1. Executive Summary

This project aimed to develop a robust machine learning system to predict hotel reservation cancellations. By analyzing booking data, we identified key drivers of cancellations and built high-performance classification models. The final Random Forest and XGBoost models demonstrate strong predictive capability, enabling the hotel to optimize revenue through dynamic overbooking and targeted retention strategies.

## 2. Methodology

### 2.1 Data Exploration & Preprocessing

- **Data Source**: Hotel reservation dataset containing booking details, guest demographics, and status.
- **Key Features Analyzed**: Lead Time, Market Segment, Average Price, Special Requests.
- **Preprocessing**:
  - Categorical variables (`meal_type`, `market_segment`) were One-Hot Encoded.
  - Numerical variables were scaled using `StandardScaler` to ensure model stability.
  - Target variable `booking_status` was encoded (Canceled=1, Not Canceled=0).

### 2.2 Model Development

We implemented and compared three distinct algorithms:

1.  **Logistic Regression**: Established a baseline for linear relationships.
2.  **Random Forest**: Captures complex non-linear interactions and feature importance.
3.  **XGBoost**: Advanced gradient boosting for maximum predictive performance.

### 2.3 Evaluation Metrics

Models were evaluated on:

- **Precision/Recall**: To balance false positives (predicting cancel when guest arrives) and false negatives (predicting arrival when guest cancels).
- **ROC-AUC**: To assess overall ranking capability.

## 3. Key Findings

### 3.1 Drivers of Cancellation

Based on SHAP analysis and feature importance:

1.  **Lead Time**: Identifying that bookings made far in advance are significantly more likely to cancel.
2.  **Market Segment**: Online bookings show higher cancellation volatility compared to Corporate bookings.
3.  **Special Requests**: Guests making special requests are more committed and less likely to cancel.
4.  **Price Sensitivity**: Higher average room prices correlate with slightly increased cancellation probability.

### 3.2 Model Performance

- **Random Forest** and **XGBoost** outperformed the baseline Logistic Regression.
- The models successfully identify high-risk bookings with high accuracy.
- (See `reports/figures` and `notebooks` for detailed confusion matrices and ROC curves).

## 4. Business Recommendations

### 4.1 Dynamic Pricing & Overbooking

- **Overbooking Strategy**: For dates with a high predicted cancellation volume (e.g., >20%), the hotel can safely overbook by a calculated margin (e.g., 5-10%) to maximize occupancy without walking guests.
- **Pricing**: Offer "Non-Refundable" rates to high-risk segments (e.g., long lead time online bookings) to secure revenue.

### 4.2 Retention Strategies

- **Targeted Communication**: For bookings flagged as "High Risk" but high value, trigger an automated email campaign ~1 week before cancellation deadline offering a small perk (e.g., free drink, room upgrade) to reconfirm intent.

## 5. Deliverables included in Zip

- `notebooks/Hotel_Cancellation_Analysis.ipynb`: Complete code and analysis.
- `reports/figures/`: Generated visualization plots.
- `src/`: Source code for reproducibility.
