# Hotel Reservation Cancellation Prediction

A machine learning project to predict hotel reservation cancellations and enable data-driven business strategies for revenue optimization.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Business Recommendations](#business-recommendations)
- [Results & Deliverables](#results--deliverables)
- [Technologies Used](#technologies-used)

## 🎯 Project Overview

This project develops a robust machine learning system to predict hotel reservation cancellations. By analyzing booking data and guest behavior patterns, we built high-performance classification models to identify bookings at risk of cancellation. These predictions enable hotels to optimize revenue through dynamic overbooking strategies, targeted retention campaigns, and dynamic pricing adjustments.

## 🔍 Problem Statement

Hotel cancellations result in significant revenue loss and operational inefficiencies. Without accurate predictions, hotels struggle to:
- Optimize room inventory management
- Plan staffing and resources effectively
- Implement targeted retention strategies
- Maximize revenue through dynamic pricing

This project addresses these challenges by building predictive models that accurately forecast cancellation risk.

## 🎪 Project Objectives

1. **Exploratory Data Analysis**: Understand booking data characteristics and identify cancellation drivers
2. **Preprocessing & Feature Engineering**: Prepare data for machine learning models
3. **Model Development**: Build and compare multiple classification algorithms
4. **Model Evaluation**: Assess performance using relevant metrics
5. **Business Insights**: Provide actionable recommendations for revenue optimization
6. **Deployment Ready**: Create an interactive web application for real-time predictions

## 📁 Project Structure

```
Hotel_Cancellation_Project_Submission/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── hotel_reservations.csv        # Hotel booking dataset
├── models/
│   ├── rf_model.pkl                  # Trained Random Forest model
│   ├── xgb_model.pkl                 # Trained XGBoost model
│   └── logistic_model.pkl            # Trained Logistic Regression baseline
├── notebooks/
│   └── Hotel_Cancellation_Analysis.ipynb  # Complete analysis & modeling notebook
├── reports/
│   ├── final_report.md               # Comprehensive project report
│   └── figures/                      # Generated visualizations (plots, charts)
└── src/
    ├── __init__.py                   # Package initialization
    ├── app.py                        # Streamlit web application
    ├── create_notebook.py            # Notebook generation utilities
    ├── data_loader.py                # Data loading & preprocessing
    ├── eda_analysis.py               # Exploratory data analysis
    ├── modeling.py                   # Model training & evaluation
    └── package_submission.py         # Submission packaging utilities
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required libraries:
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning models
- xgboost - Advanced gradient boosting
- matplotlib, seaborn - Visualization
- shap - Model explainability
- streamlit - Web application framework
- jupyter - Notebook environment

### Step 2: Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, streamlit; print('All dependencies installed successfully!')"
```

## 💻 Usage

### Option 1: Run the Jupyter Notebook (Recommended for Analysis)

```bash
jupyter notebook notebooks/Hotel_Cancellation_Analysis.ipynb
```

This provides the complete analysis pipeline with:
- Data exploration and visualization
- Feature preprocessing and engineering
- Model training and comparison
- Detailed evaluation and insights

### Option 2: Use the Interactive Streamlit App (Real-time Predictions)

```bash
streamlit run src/app.py
```

The app provides:
- User-friendly interface for entering reservation details
- Real-time cancellation risk predictions
- Probability scores and risk classification
- Feature explanations

**Input Features:**
- Lead Time (days in advance)
- Average Price per Room ($)
- Number of Adults & Children
- Week Nights & Weekend Nights
- Market Segment (Online, Offline, Corporate, etc.)
- Meal Plan
- Room Type
- Radio Car Parking (Yes/No)
- Special Requests

### Option 3: Use Python Scripts Directly

```bash
# Load and preprocess data
python src/data_loader.py

# Run exploratory analysis
python src/eda_analysis.py

# Train and evaluate models
python src/modeling.py
```

## 📊 Key Findings

### Drivers of Cancellation

Based on analysis and SHAP feature importance:

1. **Lead Time** 🔴 (Strongest Driver)
   - Bookings made far in advance (>60 days) show significantly higher cancellation rates
   - Guests booking months ahead are less committed

2. **Market Segment** 📱
   - Online bookings exhibit higher cancellation volatility
   - Corporate bookings have lower, more stable cancellation rates
   - Complementary bookings show unique patterns

3. **Special Requests** ✅ (Protective Factor)
   - Guests requesting room modifications/amenities are more committed
   - Special requests reduce cancellation probability
   - Indicates higher booking intent

4. **Price Sensitivity** 💰
   - Higher average room prices show slightly increased cancellation probability
   - Premium guests may have more flexibility to cancel
   - Price-to-value perception plays a role

5. **Guest Composition** 👥
   - Families with children show different cancellation patterns
   - Single travelers vs. groups behave differently

## 🏆 Model Performance

### Models Compared

1. **Logistic Regression**
   - Role: Baseline linear model
   - Quick, interpretable, good for understanding linear relationships

2. **Random Forest** ⭐ (Recommended)
   - Captures non-linear patterns and feature interactions
   - Strong performance with explainability
   - Excellent precision-recall balance

3. **XGBoost** ⭐⭐ (Best Performance)
   - Advanced gradient boosting algorithm
   - Highest accuracy and ROC-AUC scores
   - Best at capturing complex patterns

### Evaluation Metrics

Models evaluated on:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Minimizing false positives (predicting cancel when guest arrives)
- **Recall**: Minimizing false negatives (missing actual cancellations)
- **ROC-AUC**: Overall ranking capability of the model
- **F1-Score**: Balanced metric for imbalanced data

### Results Summary

- Random Forest and XGBoost significantly outperformed baseline Logistic Regression
- Models successfully identify high-risk bookings with high accuracy
- Detailed confusion matrices and ROC curves available in `reports/figures/`
- Trade-offs between precision and recall are documented for business use

## 💡 Business Recommendations

### 1. Dynamic Overbooking Strategy

- **Implementation**: For dates with predicted cancellation volume >20%, safely overbook by 5-10%
- **Benefit**: Maximize occupancy without walking guests
- **Risk Mitigation**: Reserve nearby hotel partnerships for guest accommodation
- **Expected Impact**: 5-15% revenue increase

### 2. Targeted Retention Campaigns

- **High Risk, High Value Bookings**: Trigger automated personalized email ~7 days before cancellation deadline
  - Offer incentives: free room upgrade, welcome drink, spa credit
  - Personalized messaging based on booking characteristics
  - Track reconfirmation rates

- **Segmented Approach**:
  - Online long lead-time bookings: Early confirmation email
  - Low-price bookings: Highlight value add-ons
  - Corporate bookings: Loyalty rewards

### 3. Pricing Strategy Adjustments

- **Non-Refundable Rates**: Market to high-risk segments (long lead-time, online bookings)
  - Offer 10-15% discount for non-refundable bookings
  - Secure revenue from high-cancellation-risk segments

- **Dynamic Pricing**: Adjust rates based on predicted cancellation probability
  - Higher prices for stable low-cancellation segments
  - Competitive pricing for volatile segments

### 4. Operational Planning

- **Staffing & Resource Allocation**: 
  - Predict occupancy based on cancellation forecasts
  - Adjust housekeeping and front desk staffing
  - Plan maintenance windows during predicted low-occupancy periods

- **Marketing & Sales**:
  - Prioritize retention efforts for high-value, high-risk guests
  - Allocate marketing budget based on segment reliability

## 📦 Results & Deliverables

### Generated Files

- **`notebooks/Hotel_Cancellation_Analysis.ipynb`**
  - Complete code and step-by-step analysis
  - Data exploration and visualization
  - Model training and evaluation
  - SHAP explainability analysis

- **`reports/final_report.md`**
  - Comprehensive executive summary
  - Detailed methodology documentation
  - Key findings and business recommendations
  - Model performance comparison

- **`reports/figures/`**
  - Data distribution plots
  - Correlation matrices
  - Feature importance visualizations
  - Confusion matrices
  - ROC curves
  - SHAP plots for model explainability

- **`models/`**
  - `rf_model.pkl` - Random Forest model (recommended)
  - `xgb_model.pkl` - XGBoost model (best performance)
  - `logistic_model.pkl` - Logistic Regression baseline

- **`src/app.py`**
  - Interactive Streamlit web application
  - Real-time prediction interface
  - User-friendly prediction tool

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning models & preprocessing |
| **XGBoost** | Advanced gradient boosting |
| **SHAP** | Model explainability & feature importance |
| **Matplotlib & Seaborn** | Data visualization |
| **Streamlit** | Web application framework |
| **Jupyter** | Interactive notebook environment |

## 📈 Key Metrics & Performance

### Dataset Characteristics
- Total Records: Hotel reservation bookings
- Prediction Target: Booking cancellation (Yes/No)
- Features: 15+ booking and guest characteristics
- Class Balance: Analyzed for imbalance handling

### Model Selection Rationale
- **Chosen Model**: Random Forest or XGBoost
- **Reasoning**: Superior predictive power, feature importance insights, good generalization
- **Production Ready**: Yes - models saved and ready for deployment

## 🔗 File Descriptions

### Source Code Files

1. **`data_loader.py`**: Loads CSV data, handles missing values, encodes categorical variables
2. **`eda_analysis.py`**: Statistical analysis, visualizations, correlation analysis
3. **`modeling.py`**: Model training, hyperparameter tuning, performance evaluation
4. **`app.py`**: Streamlit application for interactive predictions
5. **`create_notebook.py`**: Utilities for notebook generation and documentation

## 📝 Notes

- All models are pre-trained and saved in `/models` directory
- Data preprocessing steps are reproducible and documented
- Results are sensitive to data quality - ensure clean input when deploying
- Regular model retraining recommended as new booking data becomes available

## 🤝 Contributing

To extend or improve this project:

1. Update models with new data: Run `modeling.py` with new data
2. Add new features: Modify `data_loader.py` and rerun analysis
3. Improve app UI: Edit `app.py`
4. Document changes in this README

## 📧 Contact & Support

For questions or feedback about this project, please refer to the documentation in:
- `reports/final_report.md` - Comprehensive analysis report
- `notebooks/Hotel_Cancellation_Analysis.ipynb` - Detailed code and explanations

---

**Project Status**: ✅ Complete  
**Last Updated**: February 2026  
**Model Readiness**: Production Ready

