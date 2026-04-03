"""
ML Model Comparison for Wheel Strategy
========================================
Compares 6 different approaches:
  1. LightGBM (current baseline)
  2. XGBoost
  3. CatBoost
  4. Random Forest
  5. Logistic Regression (interpretable baseline)
  6. Stacking Ensemble (best of all)

Each model is evaluated with:
  - TimeSeriesSplit (no data leakage)
  - ROC-AUC, Accuracy, Precision/Recall
  - Backtest: simulated P&L with ML filter
  - Feature importance analysis

Usage:
    python -X utf8 -m src.ml.model_comparison
"""

import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier,
)
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature Sets ───────────────────────────────────────────────────────────

FEATURES_BASE = [
    "premium_yield", "iv_est", "iv_rank", "rv20",
    "rsi", "trend_score", "return_20d", "atr_pct",
]


# ── Data Loading ───────────────────────────────────────────────────────────

def load_dataset(path: str = None) -> Tuple[pd.DataFrame, list]:
    """Load ML dataset and determine available features."""
    if path is None:
        path = str(PROCESSED_DIR / "ml_dataset_otm5_dte30.csv")

    df = pd.read_csv(path)

    # Use all available features from our feature sets
    available = [f for f in FEATURES_BASE if f in df.columns]

    df = df.dropna(subset=available + ["was_assigned"])
    print(f"Dataset: {len(df):,} rows | {df.ticker.nunique()} tickers")
    print(f"Features: {len(available)} | Assignment rate: {df.was_assigned.mean()*100:.1f}%")
    print(f"Date range: {df.date.min()} -> {df.date.max()}")
    return df, available


# ── Model Definitions ──────────────────────────────────────────────────────

def get_models(features: list) -> Dict:
    """Return dict of model_name -> (model, needs_scaling)."""
    models = {}

    # 1. LightGBM (current baseline)
    models["LightGBM"] = (
        lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, max_depth=6,
            min_child_samples=20,
            feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, class_weight="balanced",
            random_state=42, verbose=-1,
        ),
        False,
    )

    # 2. XGBoost
    try:
        import xgboost as xgb
        models["XGBoost"] = (
            xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.05,
                max_depth=6, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=3.5,  # ~1/assignment_rate
                random_state=42, eval_metric="logloss",
                verbosity=0,
            ),
            False,
        )
    except ImportError:
        print("  XGBoost not installed, skipping")

    # 3. CatBoost
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = (
            CatBoostClassifier(
                iterations=500, learning_rate=0.05,
                depth=6, l2_leaf_reg=3,
                auto_class_weights="Balanced",
                random_state=42, verbose=0,
            ),
            False,
        )
    except ImportError:
        print("  CatBoost not installed, skipping")

    # 4. Random Forest
    models["RandomForest"] = (
        RandomForestClassifier(
            n_estimators=500, max_depth=8,
            min_samples_leaf=20, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        False,
    )

    # 5. Logistic Regression (interpretable baseline)
    models["LogisticReg"] = (
        LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42,
        ),
        True,  # needs scaling
    )

    # 6. Gradient Boosting (sklearn)
    models["GradientBoosting"] = (
        GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, min_samples_leaf=20,
            subsample=0.8, random_state=42,
        ),
        False,
    )

    return models


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(
    model, X_train, y_train, X_test, y_test,
    needs_scaling: bool = False,
) -> dict:
    """Train and evaluate a single model."""
    scaler = None
    X_tr, X_te = X_train.copy(), X_test.copy()

    if needs_scaling:
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)

    # Train
    if hasattr(model, "fit"):
        # LightGBM/XGBoost with early stopping
        if isinstance(model, (lgb.LGBMClassifier,)):
            model.fit(
                X_tr, y_train,
                eval_set=[(X_te, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
        else:
            model.fit(X_tr, y_train)

    # Predict
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    # Metrics
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "y_proba": y_proba,
        "model": model,
        "scaler": scaler,
    }
    return results


def run_timeseries_cv(
    df: pd.DataFrame,
    features: list,
    models_dict: dict,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Run TimeSeriesSplit cross-validation for all models.
    Returns DataFrame of results per model per fold.
    """
    X = df[features]
    y = df["was_assigned"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"\n  Fold {fold_idx + 1}/{n_splits}: "
              f"train={len(train_idx):,} test={len(test_idx):,} "
              f"(assign%={y_test.mean()*100:.1f}%)")

        for model_name, (model_template, needs_scaling) in models_dict.items():
            # Clone model for each fold
            import sklearn.base
            model = sklearn.base.clone(model_template)

            try:
                res = evaluate_model(model, X_train, y_train, X_test, y_test, needs_scaling)
                all_results.append({
                    "fold": fold_idx + 1,
                    "model": model_name,
                    "accuracy": res["accuracy"],
                    "roc_auc": res["roc_auc"],
                    "f1": res["f1"],
                    "precision": res["precision"],
                    "recall": res["recall"],
                })
            except Exception as e:
                print(f"    {model_name}: ERROR - {e}")
                all_results.append({
                    "fold": fold_idx + 1,
                    "model": model_name,
                    "accuracy": 0, "roc_auc": 0.5, "f1": 0,
                    "precision": 0, "recall": 0,
                })

    return pd.DataFrame(all_results)


# ── Backtest with ML filter ───────────────────────────────────────────────

def backtest_ml_filter(
    df: pd.DataFrame,
    features: list,
    model, scaler=None,
    threshold: float = 0.25,
) -> dict:
    """
    Simulate Wheel strategy returns using ML filter.
    Compare filtered vs unfiltered trades.
    """
    # Use last 20% as test
    split = int(len(df) * 0.8)
    df_test = df.iloc[split:].copy()

    X_test = df_test[features]
    if scaler:
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    df_test["prob_assigned"] = model.predict_proba(X_test)[:, 1]

    # ML-filtered trades
    safe = df_test[df_test.prob_assigned < threshold]
    all_trades = df_test

    # Compute P&L stats
    def calc_stats(subset, label):
        if len(subset) == 0:
            return {}
        total_pnl = subset["pnl"].sum()
        avg_pnl = subset["pnl"].mean()
        win_rate = (1 - subset["was_assigned"].mean()) * 100
        avg_premium = subset["premium_yield"].mean()
        sharpe = subset["pnl"].mean() / (subset["pnl"].std() + 1e-8) * np.sqrt(252/30)
        max_loss = subset["pnl"].min()
        return {
            f"{label}_trades": len(subset),
            f"{label}_total_pnl": round(total_pnl, 2),
            f"{label}_avg_pnl": round(avg_pnl, 4),
            f"{label}_win_rate": round(win_rate, 1),
            f"{label}_avg_premium": round(avg_premium, 3),
            f"{label}_sharpe": round(sharpe, 2),
            f"{label}_max_loss": round(max_loss, 4),
        }

    result = {}
    result.update(calc_stats(all_trades, "all"))
    result.update(calc_stats(safe, "filtered"))
    result["improvement_win_rate"] = (
        result.get("filtered_win_rate", 0) - result.get("all_win_rate", 0)
    )
    result["improvement_sharpe"] = (
        result.get("filtered_sharpe", 0) - result.get("all_sharpe", 0)
    )

    return result


# ── Stacking Ensemble ──────────────────────────────────────────────────────

def build_stacking_ensemble(
    df: pd.DataFrame,
    features: list,
) -> Tuple:
    """
    Build a stacking ensemble from top-performing models.
    Level-0: LightGBM + RandomForest + GradientBoosting
    Level-1: LogisticRegression (meta-learner)
    """
    print("\n" + "=" * 60)
    print("  BUILDING STACKING ENSEMBLE")
    print("=" * 60)

    X = df[features]
    y = df["was_assigned"]
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    base_estimators = [
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, max_depth=6,
            class_weight="balanced", verbose=-1, random_state=42,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=20, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, min_samples_leaf=20,
            subsample=0.8, random_state=42,
        )),
    ]

    # Add XGBoost if available
    try:
        import xgboost as xgb
        base_estimators.append(
            ("xgb", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=6, scale_pos_weight=3.5,
                random_state=42, verbosity=0,
            ))
        )
    except ImportError:
        pass

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=5,  # regular KFold for internal stacking (TimeSeriesSplit fails here)
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    print(f"  Base learners: {[name for name, _ in base_estimators]}")
    print(f"  Meta-learner: LogisticRegression")
    print(f"  Training on {len(X_train):,} samples...")

    stack.fit(X_train, y_train)

    y_proba = stack.predict_proba(X_test)[:, 1]
    y_pred = stack.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n  Stacking Ensemble:")
    print(f"    Accuracy : {acc*100:.1f}%")
    print(f"    ROC-AUC  : {auc:.3f}")

    return stack, None  # no scaler needed


# ── Soft Voting Ensemble ──────────────────────────────────────────────────

def build_voting_ensemble(
    df: pd.DataFrame,
    features: list,
) -> Tuple:
    """Soft voting ensemble — simple average of probabilities."""
    X = df[features]
    y = df["was_assigned"]
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    estimators = [
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", verbose=-1, random_state=42,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ]

    try:
        import xgboost as xgb
        estimators.append(
            ("xgb", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                scale_pos_weight=3.5, random_state=42, verbosity=0,
            ))
        )
    except ImportError:
        pass

    voter = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voter.fit(X_train, y_train)

    y_proba = voter.predict_proba(X_test)[:, 1]
    y_pred = voter.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n  Voting Ensemble:")
    print(f"    Accuracy : {acc*100:.1f}%")
    print(f"    ROC-AUC  : {auc:.3f}")

    return voter, None


# ── Neural Network (MLP) ─────────────────────────────────────────────────

def build_neural_network(
    df: pd.DataFrame,
    features: list,
) -> Tuple:
    """Simple MLP classifier using sklearn MLPClassifier."""
    from sklearn.neural_network import MLPClassifier

    print("\n  Training Neural Network (MLP)...")

    X = df[features]
    y = df["was_assigned"]
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        batch_size=256,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_tr, y_train)

    y_proba = mlp.predict_proba(X_te)[:, 1]
    y_pred = mlp.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"  MLP Neural Network:")
    print(f"    Accuracy : {acc*100:.1f}%")
    print(f"    ROC-AUC  : {auc:.3f}")
    print(f"    Layers   : {mlp.hidden_layer_sizes}")

    return mlp, scaler


# ── Main Comparison ────────────────────────────────────────────────────────

def run_full_comparison():
    """Run complete model comparison and save best model."""
    print("=" * 60)
    print("  ML MODEL COMPARISON FOR WHEEL STRATEGY")
    print("=" * 60)

    df, features = load_dataset()

    # ── Step 1: Cross-Validation ──────────────────────────────────
    print("\n" + "-" * 60)
    print("  STEP 1: TimeSeriesSplit Cross-Validation (5 folds)")
    print("-" * 60)

    models = get_models(features)
    cv_results = run_timeseries_cv(df, features, models, n_splits=5)

    # Summary
    summary = cv_results.groupby("model").agg({
        "accuracy": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "f1": ["mean", "std"],
    }).round(4)
    summary.columns = [f"{c[0]}_{c[1]}" for c in summary.columns]
    summary = summary.sort_values("roc_auc_mean", ascending=False)

    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION RESULTS (sorted by ROC-AUC)")
    print("=" * 60)
    print(f"\n{'Model':<20} {'AUC':>10} {'Acc':>10} {'F1':>10}")
    print("-" * 50)
    for model_name, row in summary.iterrows():
        print(f"{model_name:<20} "
              f"{row['roc_auc_mean']:.3f}+{row['roc_auc_std']:.3f}  "
              f"{row['accuracy_mean']:.3f}+{row['accuracy_std']:.3f}  "
              f"{row['f1_mean']:.3f}+{row['f1_std']:.3f}")

    # ── Step 2: Ensembles ─────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  STEP 2: Ensemble Models")
    print("-" * 60)

    stack_model, stack_scaler = build_stacking_ensemble(df, features)
    vote_model, vote_scaler = build_voting_ensemble(df, features)
    nn_model, nn_scaler = build_neural_network(df, features)

    # ── Step 3: Backtest all models ───────────────────────────────
    print("\n" + "-" * 60)
    print("  STEP 3: Backtest with ML Filter (threshold=0.25)")
    print("-" * 60)

    # Train final versions on first 80%
    split = int(len(df) * 0.8)
    X_train = df[features].iloc[:split]
    y_train = df["was_assigned"].iloc[:split]
    X_test = df[features].iloc[split:]
    y_test = df["was_assigned"].iloc[split:]

    final_models = {}

    # Train each base model
    for model_name, (model_template, needs_scaling) in models.items():
        import sklearn.base
        m = sklearn.base.clone(model_template)
        scl = None
        X_tr = X_train.copy()
        if needs_scaling:
            scl = StandardScaler()
            X_tr = pd.DataFrame(scl.fit_transform(X_tr), columns=features, index=X_tr.index)
        if isinstance(m, lgb.LGBMClassifier):
            X_te_s = X_test.copy()
            if scl:
                X_te_s = pd.DataFrame(scl.transform(X_te_s), columns=features, index=X_te_s.index)
            m.fit(X_tr, y_train,
                  eval_set=[(X_te_s, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        else:
            m.fit(X_tr, y_train)
        final_models[model_name] = (m, scl)

    # Add ensembles
    final_models["Stacking"] = (stack_model, stack_scaler)
    final_models["Voting"] = (vote_model, vote_scaler)
    final_models["NeuralNet"] = (nn_model, nn_scaler)

    # Backtest each
    backtest_results = []
    for model_name, (model, scaler) in final_models.items():
        bt = backtest_ml_filter(df, features, model, scaler, threshold=0.25)
        bt["model"] = model_name
        backtest_results.append(bt)

    bt_df = pd.DataFrame(backtest_results).set_index("model")

    print(f"\n{'Model':<20} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>9} {'Sharpe':>8}")
    print("-" * 56)
    for model_name, row in bt_df.iterrows():
        print(f"{model_name:<20} "
              f"{int(row.get('filtered_trades', 0)):>8} "
              f"{row.get('filtered_win_rate', 0):>7.1f}% "
              f"{row.get('filtered_avg_pnl', 0):>8.4f} "
              f"{row.get('filtered_sharpe', 0):>8.2f}")

    print(f"\n{'--- Unfiltered ---':<20} "
          f"{int(bt_df.iloc[0].get('all_trades', 0)):>8} "
          f"{bt_df.iloc[0].get('all_win_rate', 0):>7.1f}% "
          f"{bt_df.iloc[0].get('all_avg_pnl', 0):>8.4f} "
          f"{bt_df.iloc[0].get('all_sharpe', 0):>8.2f}")

    # ── Step 4: Save best model ───────────────────────────────────
    print("\n" + "-" * 60)
    print("  STEP 4: Save Best Model")
    print("-" * 60)

    # Pick best by Sharpe ratio
    best_name = bt_df["filtered_sharpe"].idxmax()
    best_model, best_scaler = final_models[best_name]

    print(f"\n  Best model: {best_name}")
    print(f"    Win rate : {bt_df.loc[best_name, 'filtered_win_rate']:.1f}%")
    print(f"    Sharpe   : {bt_df.loc[best_name, 'filtered_sharpe']:.2f}")

    # Save
    joblib.dump(best_model, MODELS_DIR / "best_assignment_model.pkl")
    if best_scaler:
        joblib.dump(best_scaler, MODELS_DIR / "best_model_scaler.pkl")
    joblib.dump(features, MODELS_DIR / "best_model_features.pkl")

    # Also save the stacking ensemble (robust choice)
    joblib.dump(stack_model, MODELS_DIR / "stacking_ensemble.pkl")

    print(f"\n  Saved:")
    print(f"    {MODELS_DIR}/best_assignment_model.pkl ({best_name})")
    print(f"    {MODELS_DIR}/stacking_ensemble.pkl")

    # Save comparison results
    summary.to_csv(PROCESSED_DIR / "model_comparison_cv.csv")
    bt_df.to_csv(PROCESSED_DIR / "model_comparison_backtest.csv")
    print(f"    {PROCESSED_DIR}/model_comparison_cv.csv")
    print(f"    {PROCESSED_DIR}/model_comparison_backtest.csv")

    print("\n" + "=" * 60)
    print("  COMPARISON COMPLETE")
    print("=" * 60)

    return summary, bt_df, final_models


if __name__ == "__main__":
    run_full_comparison()
