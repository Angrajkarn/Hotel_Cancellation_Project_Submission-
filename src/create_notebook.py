import nbformat as nbf
import os
import sys

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    text_intro = """# Hotel Reservation Cancellation Prediction - Advanced Analysis
    
## Project Overview
This notebook presents an advanced machine learning solution to predict hotel reservation cancellations.
We utilize synthetic data mimicking real-world hotel booking scenarios to build robust classification models.

## Methodology
1. Data Loading & Generation
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing (Scaling, Encoding)
4. Model Training (Logistic Regression, Random Forest, XGBoost)
5. Evaluation & SHAP Explanation
"""
    
    code_load = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import xgboost as xgb
import shap

# Set plotting style
sns.set(style="whitegrid")

# Load Data
data_path = '../data/hotel_reservations.csv'
df = pd.read_csv(data_path)
print(f"Data Loaded: {df.shape}")
df.head()
"""

    text_eda = """## Exploratory Data Analysis
We analyze feature distributions and correlations to understand factors driving cancellations.
"""
    
    code_eda = """
# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='booking_status', hue='booking_status', data=df, palette='viridis', legend=False)
plt.title('Distribution of Booking Status')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# Lead Time vs Cancellation
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='lead_time', hue='booking_status', kde=True, element="step", palette='viridis')
plt.title('Lead Time Distribution by Booking Status')
plt.show()

# Market Segment Impact
plt.figure(figsize=(10, 6))
sns.countplot(x='market_segment_type', hue='booking_status', data=df, palette='viridis')
plt.title('Cancellation Rates by Market Segment')
plt.xticks(rotation=45)
plt.show()
"""

    text_modeling = """## Model Development
We implement a pipeline with preprocessing (OneHotEncoding, StandardScaler) and train Logistic Regression, Random Forest, and XGBoost models.
"""
    
    code_preprocess = """
# Feature Engineering & Preprocessing
df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
df['total_guests'] = df['no_of_adults'] + df['no_of_children']

X = df.drop(['booking_status', 'Booking_ID', 'arrival_date', 'arrival_year'], axis=1)
y = df['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing Pipeline
cat_cols = ['type_of_meal_plan', 'market_segment_type', 'room_type_reserved']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

print("Preprocessing Pipeline Defined")
"""

    code_training = """
# Define Models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"--- {name} Results ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("-" * 30)
    
    trained_models[name] = pipeline
"""

    text_shap = """## Model Interpretability
Using SHAP (SHapley Additive exPlanations) to understand feature importance for the Random Forest model.
"""

    code_shap = """
# SHAP Analysis on Random Forest
rf_pipeline = trained_models['RandomForest']
preprocessor = rf_pipeline.named_steps['preprocessor']
rf_model = rf_pipeline.named_steps['classifier']

# Transform X_test to get feature names
X_test_transformed = preprocessor.transform(X_test)
# Convert to dense if sparse (SHAP prefers dense arrays)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

cat_encoder = preprocessor.named_transformers_['cat']
feature_names = num_cols + list(cat_encoder.get_feature_names_out(cat_cols))

# SHAP Explainer
explainer = shap.TreeExplainer(rf_model)
# Optimize: Use only the first 500 samples for speed
X_test_sample = X_test_transformed[:500]
shap_values = explainer.shap_values(X_test_sample, check_additivity=False)

# Visualization
# Check if shap_values is a list (binary classification usually returns [class0, class1])
if isinstance(shap_values, list):
    vals = shap_values[1]
else:
    vals = shap_values

shap.summary_plot(vals, X_test_sample, feature_names=feature_names)
"""

    code_roi = """
# --- ROI / Business Value Analysis ---

def calculate_monthly_revenue_impact(y_test, y_pred, avg_room_price=100, overbooking_limit=0.1):
    # Assumptions
    revenue_per_night = avg_room_price
    cost_of_walking_guest = avg_room_price * 2  # Cost if we overbook and guest arrives (relocation cost)
    
    # Financials WITHOUT Model (No overbooking policy)
    actual_stays = (y_test == 0).sum()
    revenue_baseline = actual_stays * revenue_per_night
    
    # Financials WITH Model
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Revenue from normal stays (TN)
    revenue_model = tn * revenue_per_night
    
    # Revenue from re-selling cancelled rooms (TP)
    # Assume we can re-sell 80% of rooms we CORRECTLY predicted would cancel
    resell_rate = 0.8
    revenue_from_resell = (tp * resell_rate) * revenue_per_night
    
    # Cost of mistakes (FP)
    cost_errors = fp * cost_of_walking_guest
    
    net_revenue_model = revenue_model + revenue_from_resell - cost_errors
    
    print(f"--- Financial Impact Analysis (Test Set) ---")
    print(f"Baseline Revenue (No Model): ${revenue_baseline:,.2f}")
    print(f"Model-Driven Revenue:        ${net_revenue_model:,.2f}")
    print(f"---------------------------------------------")
    profit = net_revenue_model - revenue_baseline
    print(f"Net Profit using AI Model:   ${profit:,.2f}")
    
    # Visualization
    labels = ['Baseline Revenue', 'Model-Driven Revenue']
    values = [revenue_baseline, net_revenue_model]
    colors = ['gray', 'green' if profit > 0 else 'red']
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, width=0.5)
    plt.title('Business Impact: Revenue Comparison')
    plt.ylabel('Revenue ($)')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'${height:,.0f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    plt.show()

# Calculate with average price from dataset
avg_price = df['avg_price_per_room'].mean()
calculate_monthly_revenue_impact(y_test, y_pred, avg_room_price=avg_price)
"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(text_intro),
        nbf.v4.new_code_cell(code_load),
        nbf.v4.new_markdown_cell(text_eda),
        nbf.v4.new_code_cell(code_eda),
        nbf.v4.new_markdown_cell(text_modeling),
        nbf.v4.new_code_cell(code_preprocess),
        nbf.v4.new_code_cell(code_training),
        nbf.v4.new_markdown_cell(text_shap),
        nbf.v4.new_code_cell(code_shap),
        nbf.v4.new_code_cell(code_roi)
    ]
    
    output_path = os.path.join(os.getcwd(), 'hotel_cancellations', 'notebooks', 'Hotel_Cancellation_Analysis.ipynb')
    with open(output_path, 'w') as f:
        nbf.write(nb, f)

    print(f"Notebook created at {output_path}")

if __name__ == "__main__":
    create_notebook()
