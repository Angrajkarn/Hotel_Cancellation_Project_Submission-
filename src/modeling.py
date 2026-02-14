import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Try importing XGBoost and SHAP, handle if missing
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found, skipping XGBoost models.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not found, skipping SHAP analysis.")

# Ensure output directory exists
OUTPUT_DIR = r"c:\Users\Nivedita\.gemini\antigravity\playground\temporal-aurora\hotel_cancellations\reports\figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Feature Engineering
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    
    # Drop IDs and Dates (simplified for model, though month/day could be useful)
    # We will keep 'arrival_month' as it captures seasonality
    drop_cols = ['Booking_ID', 'arrival_date', 'arrival_year'] 
    df = df.drop(columns=drop_cols)
    
    X = df.drop('booking_status', axis=1)
    y = df['booking_status']
    
    # Encode Target
    le = LabelEncoder()
    y = le.fit_transform(y) # Canceled=0 or 1, check mapping
    # Usually Canceled=0, Not_Canceled=1 alphabetically?
    # Let's check: 'Canceled' comes before 'Not_Canceled'. 
    # So 0=Canceled, 1=Not_Canceled. 
    # But usually we want 1=Canceled (Positive class).
    # Let's force mapping if needed, or just allow it and interpret later.
    # To be safe:
    y = df['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)
    
    return X, y

def build_pipeline(model):
    # Categorical and Numerical columns
    cat_cols = ['type_of_meal_plan', 'market_segment_type', 'room_type_reserved']
    num_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                'required_car_parking_space', 'lead_time', 'arrival_month', 
                'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 
                'avg_price_per_room', 'no_of_special_requests', 'total_nights', 'total_guests']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    return pipeline

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"--- {name} Results ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{name}.png'))
    plt.close()

def run_modeling(filepath):
    X, y = load_and_preprocess_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        evaluate_model(name, pipeline, X_test, y_test)
        trained_models[name] = pipeline
        
    # Save the Random Forest model for Streamlit App
    model_dir = os.path.join(os.path.dirname(OUTPUT_DIR), '..', 'models') # hotel_cancellations/models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    rf_model_path = os.path.join(model_dir, 'rf_model.pkl')
    joblib.dump(trained_models['RandomForest'], rf_model_path)
    print(f"Model saved to {rf_model_path}")

    # SHAP Analysis on Random Forest (Advanced)
    if HAS_SHAP:
        print("Running SHAP analysis on Random Forest...")
        rf_pipeline = trained_models['RandomForest']
        preprocessor = rf_pipeline.named_steps['preprocessor']
        rf_model = rf_pipeline.named_steps['classifier']
        
        # Transform X_test to get feature names
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names from OneHotEncoder
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_cols = ['type_of_meal_plan', 'market_segment_type', 'room_type_reserved']
        num_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                'required_car_parking_space', 'lead_time', 'arrival_month', 
                'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 
                'avg_price_per_room', 'no_of_special_requests', 'total_nights', 'total_guests']
        
        feature_names = num_cols + list(cat_encoder.get_feature_names_out(cat_cols))
        
        # SHAP Explainer (TreeExplainer is efficient for RF)
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test_transformed)
        
        # Summary Plot
        plt.figure()
        # shap_values[1] for positive class (Cancellation)
        shap.summary_plot(shap_values[1], X_test_transformed, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_rf.png'))
        plt.close()

if __name__ == "__main__":
    run_modeling(r"c:\Users\Nivedita\.gemini\antigravity\playground\temporal-aurora\hotel_cancellations\data\hotel_reservations.csv")
