import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Ensure src is in path if running from root
sys.path.append(os.path.join(os.getcwd(), 'hotel_cancellations', 'src'))

def perform_eda(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    # Set style
    sns.set(style="whitegrid")
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='booking_status', hue='booking_status', data=df, palette='viridis', legend=False)
    plt.title('Distribution of Booking Status')
    plt.savefig(os.path.join(output_dir, 'booking_status_dist.png'))
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f") # annot=False to avoid clutter if many vars
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Lead Time vs Cancellation
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='lead_time', hue='booking_status', kde=True, element="step", palette='viridis')
    plt.title('Lead Time Distribution by Booking Status')
    plt.savefig(os.path.join(output_dir, 'lead_time_vs_status.png'))
    plt.close()
    
    # 4. Market Segment Impact
    plt.figure(figsize=(10, 6))
    sns.countplot(x='market_segment_type', hue='booking_status', data=df, palette='viridis')
    plt.title('Cancellation Rates by Market Segment')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'market_segment_impact.png'))
    plt.close()
    
    # 5. Price vs Cancellation
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='booking_status', y='avg_price_per_room', hue='booking_status', data=df, palette='viridis', legend=False)
    plt.title('Average Price per Room vs Booking Status')
    plt.savefig(os.path.join(output_dir, 'price_vs_status.png'))
    plt.close()

    # 6. Guest Composition (Family vs Solo/Couple)
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='total_guests', hue='booking_status', multiple="stack", palette='viridis')
    plt.title('Total Guests vs Cancellation')
    plt.savefig(os.path.join(output_dir, 'guest_composition_impact.png'))
    plt.close()

    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    perform_eda(
        r"c:\Users\Nivedita\.gemini\antigravity\playground\temporal-aurora\hotel_cancellations\data\hotel_reservations.csv",
        r"c:\Users\Nivedita\.gemini\antigravity\playground\temporal-aurora\hotel_cancellations\reports\figures"
    )
