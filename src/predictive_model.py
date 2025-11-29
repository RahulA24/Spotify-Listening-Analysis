import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import Tuple, List

def train_skip_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, float, List[str]]:
    """
    Train a Random Forest model to predict whether a track will be skipped.

    The function is flexible enough to work with both Standard and Extended 
    Spotify history. If extra fields like shuffle mode or reason_start are 
    available, they are automatically added to the feature set.
    """
    print("[INFO] Starting model training (Random Forest)")

    # --- Base features used for all datasets ---
    # 'ms_played' is intentionally left out to avoid leakage.
    features = ["hour", "day_of_week", "is_weekend"]

    # --- Optional features (only added if they exist) ---
    # Extended History exports sometimes include these.
    optional_cols = [
        "reason_start_encoded",
        "shuffle_feature",
        "is_mobile",
        "conn_country_encoded"
    ]

    for col in optional_cols:
        if col in df.columns:
            features.append(col)

    print(f"[INFO] Using features: {features}")

    # --- Train/Test split ---
    X = df[features]
    y = df["is_skipped"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Model training ---
    # max_depth=12 provides a good balance between learning detail and avoiding overfitting.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"[SUCCESS] Training complete. AUC: {auc_score:.2f}")

    return model, auc_score, features
