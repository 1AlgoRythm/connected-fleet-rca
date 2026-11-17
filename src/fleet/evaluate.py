"""Fit the three detectors on TRAIN devices, judge them on TEST devices.

Split is by device (never by row): no device contributes rows to both sides.
Thresholds are calibrated on training devices only, to a common budget:
at most 5 % of healthy training devices may ever raise an alert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from . import config as C
from .detectors import (GradientBoostingDet, IsolationForestDet, RuleBaseline, alert_days,
                        calibrate_threshold, oof_scores)

FAULT_LABELS = C.MODES   # row/device label vocabulary, healthy first


def split_devices(truth: pd.DataFrame, test_frac: float = 0.4, seed: int = 0):
    tr, te = train_test_split(truth["device_id"], test_size=test_frac, random_state=seed,
                              stratify=truth["mode"])
    assert not set(tr) & set(te), "device leaked across the split"
    return sorted(tr), sorted(te)


def _verdicts(rows: pd.DataFrame, flagged: np.ndarray, kind: str, scored: pd.DataFrame,
              proba: pd.DataFrame | None) -> pd.DataFrame:
    """One line per device: alert day and predicted mode ('healthy' if it never alerted)."""
    alerts = alert_days(rows, flagged)
    out = []
    for dev, day0 in alerts.items():
        if np.isnan(day0):
            out.append((dev, np.nan, "healthy", None))
            continue
        after = (rows["device_id"] == dev).to_numpy() & (rows["day"] >= day0).to_numpy()
        if kind == "rule":
            m = scored.loc[after & flagged, "mode"]
            verdict = m.mode().iat[0] if len(m) else scored.loc[after, "mode"].mode().iat[0]
        elif kind == "gb":
            verdict = proba.loc[after].drop(columns="healthy").mean().idxmax()
        else:
            verdict = "anomalous"
        at_alert = (rows["device_id"] == dev).to_numpy() & (rows["day"] == day0).to_numpy()
        first = proba.loc[at_alert].drop(columns="healthy").iloc[0].idxmax() if kind == "gb" else None
        out.append((dev, day0, verdict, first))
    return pd.DataFrame(out, columns=["device_id", "alert_day", "verdict", "mode_at_first_alert"])


def fit_and_judge(rows: pd.DataFrame, truth: pd.DataFrame, train_ids: list[str], test_ids: list[str],
                  gb_features: list[str] | None = None, shuffle_labels: bool = False, seed: int = 0,
                  gb_only: bool = False) -> dict:
    """Returns {'rule': verdicts, 'iforest': verdicts, 'gb': verdicts, 'gb_model', 'gb_tau', ...}."""
    train = rows[rows["device_id"].isin(train_ids)].reset_index(drop=True)
    test = rows[rows["device_id"].isin(test_ids)].reset_index(drop=True)
    dev_healthy = (truth.set_index("device_id")["mode"].loc[train_ids] == "healthy")
    labels = train["row_label"]
    if shuffle_labels:   # control: every training device gets somebody else's fault history
        rng = np.random.default_rng(seed)
        t = truth.set_index("device_id")
        donor = dict(zip(train_ids, rng.permutation(train_ids)))
        mode = train["device_id"].map(lambda d: t.loc[donor[d], "mode"])
        onset = train["device_id"].map(lambda d: t.loc[donor[d], "onset_day"]).astype(float)
        labels = pd.Series(np.where((mode != "healthy") & (train["day"] >= onset), mode, "healthy"),
                           index=train.index)
        dev_healthy = pd.Series({d: t.loc[donor[d], "mode"] == "healthy" for d in train_ids})
    res: dict = {"train_devices": len(train_ids), "test_devices": len(test_ids)}

    if not gb_only:
        # --- rule baseline
        rule = RuleBaseline().fit(train, labels == "healthy")
        s_tr, s_te = rule.score(train), rule.score(test)
        tau = calibrate_threshold(train, s_tr["score"], dev_healthy)
        res["rule_tau"] = tau
        res["rule"] = _verdicts(test, (s_te["score"] >= tau).to_numpy(), "rule", s_te, None)

        # --- isolation forest (labels never used for fitting)
        iso = IsolationForestDet(seed).fit(train)
        s_tr, s_te = iso.score(train), iso.score(test)
        tau = calibrate_threshold(train, s_tr["score"], dev_healthy)
        res["iforest_tau"] = tau
        res["iforest"] = _verdicts(test, (s_te["score"] >= tau).to_numpy(), "iforest", s_te, None)

    # --- gradient boosting classifier
    gb = GradientBoostingDet(gb_features, seed).fit(train, labels)
    oof = oof_scores(train, labels, gb_features, seed=seed)
    tau = calibrate_threshold(train, oof, dev_healthy)
    p_te = gb.proba(test)
    s_te = gb.score(test)
    res["gb_tau"] = tau
    res["gb"] = _verdicts(test, (s_te["score"] >= tau).to_numpy(), "gb", s_te, p_te)
    res.update(gb_model=gb, test_rows=test, train_rows=train, test_proba=p_te)
    return res


# ----------------------------------------------------------------------- metrics
def _prf(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                  "f1": round(float(f[i]), 4), "support": int(s[i])} for i, lab in enumerate(labels)}


def lead_stats(v: pd.DataFrame, truth: pd.DataFrame) -> dict:
    t = truth.set_index("device_id")
    out = {}
    for mode in C.DEGRADING_MODES:
        d = v[v["device_id"].map(t["mode"]) == mode].copy()
        fail = d["device_id"].map(t["hard_failure_day"])
        lead = (fail - d["alert_day"]).dropna()
        early = lead[lead > 0]
        out[mode] = {"devices": int(len(d)), "alerted": int(d["alert_day"].notna().sum()),
                     "alerted_before_hard_failure": int(len(early)),
                     "lead_days_median": float(early.median()) if len(early) else None,
                     "lead_days_p90": float(np.percentile(early, 90)) if len(early) else None}
    allearly = pd.concat([(v["device_id"].map(t["hard_failure_day"]) - v["alert_day"])[
        v["device_id"].map(t["mode"]).isin(C.DEGRADING_MODES)]]).dropna()
    allearly = allearly[allearly > 0]
    out["all_degrading"] = {"alerted_before_hard_failure": int(len(allearly)),
                            "lead_days_median": float(allearly.median()) if len(allearly) else None,
                            "lead_days_p90": float(np.percentile(allearly, 90)) if len(allearly) else None}
    fw = v[v["device_id"].map(t["mode"]) == "firmware_regression"]
    delay = (fw["alert_day"] - fw["device_id"].map(t["onset_day"])).dropna()
    out["firmware_regression_days_from_ota_to_alert"] = {
        "devices": int(len(fw)), "alerted": int(fw["alert_day"].notna().sum()),
        "median": float(delay.median()) if len(delay) else None}
    return out


def score_detector(v: pd.DataFrame, truth: pd.DataFrame, name: str) -> dict:
    t = truth.set_index("device_id")["mode"]
    y_true = v["device_id"].map(t)
    out: dict = {}
    if name != "iforest":
        y_pred = v["verdict"]
        out["per_mode"] = _prf(y_true, y_pred, FAULT_LABELS)
        out["macro_f1_all_classes"] = round(float(np.mean([m["f1"] for m in out["per_mode"].values()])), 4)
        out["macro_f1_fault_modes"] = round(float(np.mean([out["per_mode"][m]["f1"] for m in C.FAULT_MODES])), 4)
        cm = confusion_matrix(y_true, y_pred, labels=FAULT_LABELS)
        out["confusion_matrix"] = {"labels_true_rows_pred_cols": FAULT_LABELS, "matrix": cm.tolist()}
        out["accuracy"] = round(float((y_true == y_pred).mean()), 4)
    alerted = v["alert_day"].notna()
    faulty = y_true != "healthy"
    out["detection"] = {
        "any_fault_precision": round(float((faulty & alerted).sum() / max(alerted.sum(), 1)), 4),
        "any_fault_recall": round(float((faulty & alerted).sum() / faulty.sum()), 4),
        "healthy_devices_alerted": int((~faulty & alerted).sum()),
        "healthy_devices": int((~faulty).sum()),
        "healthy_false_alarm_rate": round(float((~faulty & alerted).sum() / (~faulty).sum()), 4),
        "recall_by_mode": {m: round(float(alerted[y_true == m].mean()), 4) for m in C.FAULT_MODES},
    }
    if name == "gb":   # honest early-warning view: the mode named on the very first alert day
        hit = v[alerted & faulty.to_numpy()]
        out["mode_correct_on_first_alert_day"] = {
            "devices": int(len(hit)),
            "fraction": round(float((hit["mode_at_first_alert"] == hit["device_id"].map(t)).mean()), 4)}
    out["lead_time"] = lead_stats(v, truth)
    return out
