"""Three detectors that all turn a feature row into a 'how suspicious' score.

  RuleBaseline        fixed per-signal limits from healthy units, no ML
  IsolationForestDet  unsupervised: never sees a label while fitting
  GradientBoostingDet supervised: predicts failure mode or healthy

Shared device-level logic (alerting with a debounce, threshold calibration) is at
the bottom so all three are judged by the same rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import GroupKFold

from . import config as C
from .features import MODEL_FEATURES, RATIO_FEATURES

DEBOUNCE_ROWS = 3        # alert only after 3 consecutive flagged days (kills one-day blips)
TARGET_HEALTHY_ALERT_RATE = 0.05   # tolerated share of healthy devices that ever alert


# --------------------------------------------------------------------------- rules
# (column, direction, mode it points to). direction +1 = alarm when high, -1 = when low.
RULES = [("raw_brush_a", +1, "brush_hair_wrap"), ("raw_suction_a", -1, "clogged_filter"),
         ("raw_airflow_lps", -1, "clogged_filter"), ("raw_cap_mah", -1, "battery_degradation"),
         ("raw_sag_v", +1, "battery_degradation"), ("raw_wheel_max_a", +1, "wheel_wear"),
         ("reboot_count7", +1, "firmware_regression")]


class RuleBaseline:
    """Per-signal absolute limits at a healthy-population percentile (no per-device baseline)."""

    def fit(self, rows: pd.DataFrame, healthy: pd.Series) -> "RuleBaseline":
        self.ref = {col: np.sort(rows.loc[healthy, col].to_numpy(float)) for col, _, _ in RULES}
        return self

    def score(self, rows: pd.DataFrame) -> pd.DataFrame:
        """score = max over signals of (percentile of this value within healthy rows), in [0, 1]."""
        pct = []
        for col, sign, _ in RULES:
            ref = self.ref[col]
            p = np.searchsorted(ref, rows[col].to_numpy(float), side="left") / len(ref)
            pct.append(p if sign > 0 else 1.0 - np.searchsorted(ref, rows[col].to_numpy(float), side="right") / len(ref))
        pct = np.column_stack(pct)                                    # (rows, n_rules)
        idx = pct.argmax(axis=1)
        return pd.DataFrame({"score": pct.max(axis=1),
                             "mode": [RULES[i][2] for i in idx]}, index=rows.index)


# --------------------------------------------------------------------- isolation forest
class IsolationForestDet:
    """Fit on every training row with NO labels. High score = easy to isolate = unusual."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, rows: pd.DataFrame) -> "IsolationForestDet":
        self.model = IsolationForest(n_estimators=300, max_samples=2048, random_state=self.seed, n_jobs=-1)
        self.model.fit(rows[RATIO_FEATURES].fillna(0.0))     # (rows, 17)
        return self

    def score(self, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"score": -self.model.score_samples(rows[RATIO_FEATURES].fillna(0.0))},
                            index=rows.index)


# ----------------------------------------------------------------- gradient boosting
class GradientBoostingDet:
    def __init__(self, features: list[str] | None = None, seed: int = 0):
        self.features, self.seed = features or MODEL_FEATURES, seed

    def fit(self, rows: pd.DataFrame, labels: pd.Series) -> "GradientBoostingDet":
        self.model = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", random_state=self.seed)
        self.model.fit(rows[self.features], labels)
        self.classes = list(self.model.classes_)
        return self

    def proba(self, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict_proba(rows[self.features]), index=rows.index,
                            columns=self.classes)          # (rows, n_classes)

    def score(self, rows: pd.DataFrame) -> pd.DataFrame:
        p = self.proba(rows)
        nonhealthy = p.drop(columns="healthy")
        return pd.DataFrame({"score": 1.0 - p["healthy"], "mode": nonhealthy.idxmax(axis=1)}, index=rows.index)


def oof_scores(rows: pd.DataFrame, labels: pd.Series, features: list[str] | None, n_splits: int = 3,
               seed: int = 0) -> pd.Series:
    """Out-of-fold GB scores for training rows (device-grouped), used only to pick a threshold."""
    out = pd.Series(np.nan, index=rows.index)
    for tr, te in GroupKFold(n_splits).split(rows, groups=rows["device_id"]):
        det = GradientBoostingDet(features, seed).fit(rows.iloc[tr], labels.iloc[tr])
        out.iloc[te] = det.score(rows.iloc[te])["score"].to_numpy()
    return out


# ------------------------------------------------------------- device-level decisions
def alert_days(rows: pd.DataFrame, flagged: np.ndarray, k: int = DEBOUNCE_ROWS) -> pd.Series:
    """First day on which k consecutive eval days were all flagged, per device (NaN if never)."""
    df = pd.DataFrame({"device_id": rows["device_id"].to_numpy(), "day": rows["day"].to_numpy(),
                       "flag": flagged.astype(int)}).sort_values(["device_id", "day"])
    run = df.groupby("device_id")["flag"].transform(lambda s: s.rolling(k).sum())
    hit = df[run == k]
    return hit.groupby("device_id")["day"].min().reindex(rows["device_id"].unique())


def calibrate_threshold(rows: pd.DataFrame, scores: pd.Series, device_is_healthy: pd.Series,
                        target: float = TARGET_HEALTHY_ALERT_RATE) -> float:
    """Smallest threshold whose healthy-device alert rate is <= target (on TRAINING devices only)."""
    healthy_ids = device_is_healthy[device_is_healthy].index
    mask = rows["device_id"].isin(healthy_ids).to_numpy()
    cand = np.unique(np.quantile(scores[mask], np.linspace(0.80, 0.99999, 120)))
    for tau in cand:
        alerts = alert_days(rows[mask], (scores[mask] >= tau).to_numpy())
        if alerts.notna().mean() <= target:
            return float(tau)
    return float(cand[-1])
