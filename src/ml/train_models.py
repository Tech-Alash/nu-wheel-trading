"""
Train ML models for the Wheel strategy:
  1. Assignment Classifier   — will this put be assigned? (LightGBM)
  2. Take-Profit Regressor   — optimal day to close early (LightGBM)
  3. IV Rank Predictor       — is IV high enough to sell now? (RandomForest)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "premium_yield",   # Premium as % of strike
    "iv_est",          # Estimated IV
    "iv_rank",         # IV percentile (0-100)
    "rv20",            # 20-day realized vol
    "rsi",             # RSI-14
    "trend_score",     # 0-3 (bullish trend)
    "return_20d",      # 20-day momentum
    "atr_pct",         # Average True Range %
]


def load_data(path: str = "data/processed/ml_dataset_otm5_dte30.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + ["was_assigned", "optimal_close_day"])
    print(f"Loaded {len(df):,} rows | {df.ticker.nunique()} tickers")
    print(f"Assignment rate: {df.was_assigned.mean()*100:.1f}%")
    return df


# ── 1. Assignment Classifier ───────────────────────────────────────────────

def train_assignment_classifier(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """
    Predict: will this cash-secured put be assigned?
    Target: was_assigned (0 = expired OTM, 1 = assigned)
    """
    print("\n" + "="*55)
    print("  MODEL 1: Assignment Classifier (LightGBM)")
    print("="*55)

    X = df[FEATURES]
    y = df["was_assigned"]

    # Time-based split (train on older data, test on newer)
    split = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nAccuracy : {acc*100:.1f}%")
    print(f"ROC-AUC  : {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["OTM (win)", "Assigned"]))

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Feature Importance:")
    for feat, imp in fi.items():
        bar = "█" * int(imp / fi.max() * 20)
        print(f"  {feat:15s}: {bar} ({imp:.0f})")

    # Save
    out = MODELS_DIR / "assignment_classifier.pkl"
    joblib.dump(model, out)
    print(f"\nSaved to {out}")
    return model


# ── 2. Take-Profit Regressor ───────────────────────────────────────────────

def train_takeprofit_regressor(df: pd.DataFrame) -> lgb.LGBMRegressor:
    """
    Predict: optimal day to close put for 50% profit.
    Target: optimal_close_day (1-30)
    """
    print("\n" + "="*55)
    print("  MODEL 2: Take-Profit Day Regressor (LightGBM)")
    print("="*55)

    # Only train on OTM trades (assignment = bad signal for take-profit)
    df_otm = df[df["was_assigned"] == 0].copy()
    print(f"OTM trades only: {len(df_otm):,} examples")

    X = df_otm[FEATURES]
    y = df_otm["optimal_close_day"]

    split = int(len(df_otm) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        feature_fraction=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nMAE: {mae:.1f} days")
    print(f"Avg actual close day: {y_test.mean():.1f}")
    print(f"Avg predicted day   : {y_pred.mean():.1f}")

    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature Importance:")
    for feat, imp in fi.items():
        bar = "█" * int(imp / fi.max() * 20)
        print(f"  {feat:15s}: {bar} ({imp:.0f})")

    out = MODELS_DIR / "takeprofit_regressor.pkl"
    joblib.dump(model, out)
    print(f"\nSaved to {out}")
    return model


# ── 3. IV Rank Scorer ──────────────────────────────────────────────────────

def train_iv_rank_predictor(df: pd.DataFrame) -> RandomForestRegressor:
    """
    Predict IV rank 5 days ahead — helps decide WHEN to sell.
    High IV rank = sell now (expensive options).
    Falling IV = wait.
    """
    print("\n" + "="*55)
    print("  MODEL 3: IV Rank Predictor (RandomForest)")
    print("="*55)

    df2 = df.copy()
    df2["iv_rank_future"] = df2.groupby("ticker")["iv_rank"].shift(-5)
    df2 = df2.dropna(subset=["iv_rank_future"])

    feats = ["iv_rank", "iv_est", "rv20", "rsi", "trend_score", "return_20d"]
    X = df2[feats]
    y = df2["iv_rank_future"]

    split = int(len(df2) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"MAE: {mae:.1f} IV rank points")
    print(f"Avg actual IV rank future : {y_test.mean():.1f}")
    print(f"Avg predicted IV rank     : {y_pred.mean():.1f}")

    out = MODELS_DIR / "iv_rank_predictor.pkl"
    joblib.dump(model, out)
    print(f"\nSaved to {out}")
    return model


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    m1 = train_assignment_classifier(df)
    m2 = train_takeprofit_regressor(df)
    m3 = train_iv_rank_predictor(df)

    print("\n" + "="*55)
    print("  ALL MODELS TRAINED AND SAVED")
    print("="*55)
    print(f"  {MODELS_DIR}/assignment_classifier.pkl")
    print(f"  {MODELS_DIR}/takeprofit_regressor.pkl")
    print(f"  {MODELS_DIR}/iv_rank_predictor.pkl")
    print("\nNext step: run run_daily.py --use-ml to use models in signals")
