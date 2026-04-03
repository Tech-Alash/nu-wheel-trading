"""
ML Predictor — loads trained models and scores trade candidates.

Used by signal_generator to enhance signals with ML predictions.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")

FEATURES = [
    "premium_yield",
    "iv_est",
    "iv_rank",
    "rv20",
    "rsi",
    "trend_score",
    "return_20d",
    "atr_pct",
]


class WheelPredictor:
    """Loads and runs all 3 ML models on a candidate trade."""

    def __init__(self):
        self.assignment_model  = None
        self.takeprofit_model  = None
        self.iv_rank_model     = None
        self._loaded = False

    def load(self) -> bool:
        """Load models from disk. Returns True if all loaded OK."""
        try:
            self.assignment_model = joblib.load(MODELS_DIR / "assignment_classifier.pkl")
            self.takeprofit_model = joblib.load(MODELS_DIR / "takeprofit_regressor.pkl")
            self.iv_rank_model    = joblib.load(MODELS_DIR / "iv_rank_predictor.pkl")
            self._loaded = True
            logger.info("ML models loaded OK")
            return True
        except FileNotFoundError as e:
            logger.warning(f"ML models not found ({e}) — run src/ml/train_models.py first")
            return False

    def predict(self, candidate: dict) -> dict:
        """
        Score a single trade candidate with all ML models.

        Args:
            candidate: dict with keys matching FEATURES
                       (from screener or signal_generator)

        Returns:
            dict with ML predictions added
        """
        if not self._loaded:
            return {"ml_available": False}

        # Build feature row
        row = {}
        row["premium_yield"] = candidate.get("premium_yield_annual", 0) / 12  # Monthly
        row["iv_est"]        = candidate.get("realized_vol", 0.30) * 1.15
        row["iv_rank"]       = candidate.get("iv_rank") or 30
        row["rv20"]          = candidate.get("realized_vol", 0.30)
        row["rsi"]           = candidate.get("rsi", 50)
        row["trend_score"]   = candidate.get("trend_score", 1)
        row["return_20d"]    = candidate.get("price_momentum", 0)
        row["atr_pct"]       = candidate.get("atr_pct", 2.0)

        X = pd.DataFrame([row])[FEATURES]

        # 1. Assignment probability
        prob_assigned = float(self.assignment_model.predict_proba(X)[0][1])

        # 2. Optimal close day
        optimal_day = float(self.takeprofit_model.predict(X)[0])
        optimal_day = max(1, min(30, round(optimal_day)))

        # 3. IV rank forecast (5 days)
        iv_feats = ["iv_rank", "iv_est", "rv20", "rsi", "trend_score", "return_20d"]
        X_iv = pd.DataFrame([{f: row[f] for f in iv_feats}])
        iv_rank_future = float(self.iv_rank_model.predict(X_iv)[0])

        # IV trend: rising = good (sell now), falling = wait
        iv_trend = iv_rank_future - row["iv_rank"]
        iv_signal = "SELL NOW" if iv_trend >= -5 else "WAIT — IV falling"

        # ML composite score (0-100)
        # High if: low assignment risk + high IV rank + good premium
        ml_score = (
            (1 - prob_assigned) * 50 +          # 50% weight: low assignment risk
            min(row["iv_rank"], 100) * 0.30 +   # 30% weight: IV rank
            min(row["premium_yield"] * 10, 20)  # 20% weight: premium yield
        )

        # Interpretation
        if prob_assigned < 0.15:
            assignment_risk = "LOW"
        elif prob_assigned < 0.30:
            assignment_risk = "MEDIUM"
        else:
            assignment_risk = "HIGH"

        return {
            "ml_available":       True,
            "prob_assigned":      round(prob_assigned * 100, 1),
            "assignment_risk":    assignment_risk,
            "optimal_close_day":  optimal_day,
            "iv_rank_forecast":   round(iv_rank_future, 1),
            "iv_trend":           round(iv_trend, 1),
            "iv_signal":          iv_signal,
            "ml_score":           round(ml_score, 1),
        }

    def predict_batch(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Score a full DataFrame of candidates."""
        if not self._loaded:
            candidates["ml_score"] = 0
            candidates["prob_assigned"] = 50
            candidates["optimal_close_day"] = 15
            return candidates

        results = []
        for _, row in candidates.iterrows():
            pred = self.predict(row.to_dict())
            results.append(pred)

        pred_df = pd.DataFrame(results)
        return pd.concat(
            [candidates.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1
        )


# Singleton instance
_predictor: Optional[WheelPredictor] = None


def get_predictor() -> WheelPredictor:
    """Get (or create) the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = WheelPredictor()
        _predictor.load()
    return _predictor
