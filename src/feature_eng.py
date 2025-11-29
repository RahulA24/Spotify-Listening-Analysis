import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates temporal features and dynamically handles target variable creation
    based on available data (Standard vs. Extended History).
    """
    print("[INFO] Beginning Feature Engineering...")
    df = df.copy()
    
    # --- 1. Base Temporal Features ---
    # These work on ALL data types
    df['hour'] = df['ts'].dt.hour
    df['day_of_week'] = df['ts'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # --- 2. Smart Target Variable (is_skipped) ---
    # Logic: Prefer real data ('skipped' column), fall back to proxy if missing.
    
    if 'skipped' in df.columns:
        print("[INFO] Extended Data Detected: Using accurate 'skipped' column.")
        # Handle Boolean (True/False) or Float (1.0/0.0)
        df['is_skipped'] = df['skipped'].replace({True: 1, False: 0}).fillna(0).astype(int)
        
    elif 'ms_played' in df.columns:
        print("[INFO] Standard Data Detected: Creating proxy 'is_skipped' (< 30s play).")
        df['is_skipped'] = df['ms_played'].apply(lambda x: 1 if x < 30000 else 0)
        
    else:
        df['is_skipped'] = 0
        
    # --- 3. Extended Features (Future Proofing) ---
    
    # A. Reason Start (Context: Did you click it or did it autoplay?)
    if 'reason_start' in df.columns:
        le = LabelEncoder()
        df['reason_start_encoded'] = le.fit_transform(df['reason_start'].astype(str))
        
    # B. Shuffle Mode (Context: Random vs Sequential)
    if 'shuffle' in df.columns:
        df['shuffle_feature'] = df['shuffle'].replace({True: 1, False: 0}).fillna(0).astype(int)

    # C. Platform (Context: Mobile users skip more often)
    if 'platform' in df.columns:
        # Create a simple binary: 1 for Mobile (Android/iOS), 0 for Desktop/Web
        df['is_mobile'] = df['platform'].apply(
            lambda x: 1 if 'android' in str(x).lower() or 'ios' in str(x).lower() else 0
        )
        
    print(f"[SUCCESS] Feature Engineering Complete. Columns: {len(df.columns)}")
    return df