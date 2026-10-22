"""Escalation forecasting (supervised regression).

Predicts the change in each zone's severity over the next operational window.
A linear baseline and a gradient-boosted model are both fitted; the better
held-out model is reported alongside the persistence baseline (delta=0), so
any claimed improvement is measured, not assumed (proposal Section 7.2).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .domain import ForecastReport, ForecastResult, Scenario
from .synth import (
    FORECAST_FEATURES,
    forecast_feature_vector,
    forecast_training_data,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


class EscalationForecaster:
    """Fits on seeded data and scores zones. Deterministic for a given seed."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None
        self.model_name = ""
        self._report_stats: dict = {}

    def fit(self) -> EscalationForecaster:
        X, y = forecast_training_data(self.seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=self.seed)

        lin = LinearRegression().fit(X_tr, y_tr)
        gbr = GradientBoostingRegressor(random_state=self.seed).fit(X_tr, y_tr)

        candidates = {"linear": lin, "gradient_boosted": gbr}
        scored = {name: _rmse(y_te, m.predict(X_te)) for name, m in candidates.items()}
        self.model_name = min(scored, key=scored.get)
        self.model = candidates[self.model_name]

        pred = self.model.predict(X_te)
        persistence = np.zeros_like(y_te)  # naive: severity unchanged
        self._report_stats = {
            "rmse": _rmse(y_te, pred),
            "mae": _mae(y_te, pred),
            "persistence_rmse": _rmse(y_te, persistence),
            "persistence_mae": _mae(y_te, persistence),
            "calibration": _calibration(y_te, pred),
            "importances": self._importances(),
        }
        return self

    def _importances(self) -> dict[str, float]:
        if hasattr(self.model, "feature_importances_"):
            vals = self.model.feature_importances_
        else:  # linear: use absolute standardised-ish coefficients
            vals = np.abs(self.model.coef_)
            vals = vals / (vals.sum() or 1.0)
        return {f: float(v) for f, v in zip(FORECAST_FEATURES, vals)}

    def predict_delta(self, features: list[float]) -> float:
        return float(self.model.predict([features])[0])

    def report(self, scenario: Scenario) -> ForecastReport:
        if self.model is None:
            self.fit()
        results: list[ForecastResult] = []
        for z in scenario.zones:
            f = forecast_feature_vector(
                z.severity, z.hazard_series, z.population_density,
                z.elevation_m, z.medical_cases, z.population,
            )
            delta = self.predict_delta(f)
            results.append(ForecastResult(
                zone_id=z.id, current_severity=z.severity,
                predicted_delta=delta, persistence_delta=0.0,
                predicted_next=float(np.clip(z.severity + delta, 0.0, 1.0)),
            ))
        s = self._report_stats
        return ForecastReport(
            model=self.model_name,
            rmse=s["rmse"], mae=s["mae"],
            persistence_rmse=s["persistence_rmse"], persistence_mae=s["persistence_mae"],
            beats_persistence=s["rmse"] < s["persistence_rmse"],
            feature_importances=s["importances"],
            calibration=s["calibration"],
            forecasts=results,
        )


def _calibration(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> list[tuple[float, float]]:
    """Bin predictions and compare mean prediction to mean outcome per bin."""
    order = np.argsort(y_pred)
    yp, yt = y_pred[order], y_true[order]
    out: list[tuple[float, float]] = []
    for chunk_p, chunk_t in zip(np.array_split(yp, bins), np.array_split(yt, bins)):
        if len(chunk_p):
            out.append((float(chunk_p.mean()), float(chunk_t.mean())))
    return out
