"""Root-cause views: which signals drove a flag, the fleet Pareto, firmware grouping.

Per-device explanation = occlusion: take the flagged device's worst day, replace
one signal's features by the healthy-fleet median ("pretend this signal is normal"),
and see how far the predicted probability of its fault falls. The biggest drops are
the signals the model leaned on. It is a simple, model-agnostic explanation, not a proof of cause.
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import config as C
from .features import SIGNAL_GROUPS

# What a correct explanation should mention for each injected mode (from the simulator's physics).
EXPECTED_SIGNALS = {
    "brush_hair_wrap": {"brush_motor_current"},
    "clogged_filter": {"airflow", "suction_motor_current"},
    "battery_degradation": {"battery_capacity", "battery_resistance"},
    "wheel_wear": {"wheel_motor_currents"},
    "firmware_regression": {"reboot_errors", "firmware_version"},
}


def explain_flagged(res: dict, top_k: int = 3) -> list[dict]:
    """One dict per flagged test device: predicted mode and its top contributing signals."""
    gb, test, proba = res["gb_model"], res["test_rows"], res["test_proba"]
    healthy_ref = res["train_rows"].loc[res["train_rows"]["row_label"] == "healthy", gb.features].median()
    out = []
    for _, v in res["gb"].iterrows():
        if v["verdict"] == "healthy":
            continue
        mask = ((test["device_id"] == v["device_id"]) & (test["day"] >= v["alert_day"])).to_numpy()
        idx = proba.loc[mask, v["verdict"]].idxmax()          # the day the model is most sure
        row = test.loc[[idx], gb.features].copy()
        base_p = float(gb.proba(row)[v["verdict"]].iat[0])
        drops = {}
        for sig, cols in SIGNAL_GROUPS.items():
            cols = [c for c in cols if c in gb.features]
            if not cols:
                continue
            occluded = row.copy()
            occluded[cols] = healthy_ref[cols].to_numpy()
            drops[sig] = base_p - float(gb.proba(occluded)[v["verdict"]].iat[0])
        top = sorted(drops, key=drops.get, reverse=True)[:top_k]
        out.append({"device_id": v["device_id"], "predicted_mode": v["verdict"], "alert_day": int(v["alert_day"]),
                    "explained_day": int(test.loc[idx, "day"]), "p_mode": round(base_p, 4),
                    "top_signals": [{"signal": s, "prob_drop": round(drops[s], 4)} for s in top]})
    return out


def explanation_hit_rate(expl: list[dict], truth: pd.DataFrame) -> dict:
    """Among devices whose predicted mode is right, how often a physically-expected signal is in the top 3."""
    t = truth.set_index("device_id")["mode"]
    hits = {m: [0, 0] for m in EXPECTED_SIGNALS}
    for e in expl:
        true = t[e["device_id"]]
        if e["predicted_mode"] != true or true not in EXPECTED_SIGNALS:
            continue
        hits[true][1] += 1
        if EXPECTED_SIGNALS[true] & {s["signal"] for s in e["top_signals"]}:
            hits[true][0] += 1
    out = {m: {"hit": h, "of": n, "rate": round(h / n, 4) if n else None} for m, (h, n) in hits.items()}
    tot_h, tot_n = sum(h for h, _ in hits.values()), sum(n for _, n in hits.values())
    out["overall"] = {"hit": tot_h, "of": tot_n, "rate": round(tot_h / tot_n, 4) if tot_n else None}
    return out


def permutation_importance_groups(res: dict, n_repeats: int = 3, max_rows: int = 8000, seed: int = 0) -> dict:
    """Drop in balanced accuracy (row level, test devices) when one signal group is shuffled."""
    gb, test = res["gb_model"], res["test_rows"]
    sub = test.sample(min(max_rows, len(test)), random_state=seed)
    y = sub["row_label"]
    base = balanced_accuracy_score(y, gb.model.predict(sub[gb.features]))
    rng = np.random.default_rng(seed)
    out = {}
    for sig, cols in SIGNAL_GROUPS.items():
        cols = [c for c in cols if c in gb.features]
        drops = []
        for _ in range(n_repeats):
            shuffled = sub.copy()
            shuffled[cols] = sub[cols].to_numpy()[rng.permutation(len(sub))]   # shuffle the group jointly
            drops.append(base - balanced_accuracy_score(y, gb.model.predict(shuffled[gb.features])))
        out[sig] = round(float(np.mean(drops)), 4)
    return {"baseline_row_balanced_accuracy": round(float(base), 4),
            "balanced_accuracy_drop_by_signal": dict(sorted(out.items(), key=lambda kv: -kv[1]))}


def pareto(verdicts: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Fleet-level Pareto of predicted modes among flagged devices, next to the injected truth."""
    flagged = verdicts[verdicts["verdict"] != "healthy"]
    counts = flagged["verdict"].value_counts()
    cum = counts.cumsum() / counts.sum()
    true_counts = truth[truth["mode"] != "healthy"]["mode"].value_counts()
    return {"flagged_devices": int(len(flagged)), "devices": int(len(verdicts)),
            "predicted": [{"mode": m, "devices": int(c), "share": round(float(c / counts.sum()), 4),
                           "cumulative_share": round(float(cum[m]), 4)} for m, c in counts.items()],
            "injected_truth_devices": {m: int(c) for m, c in true_counts.items()}}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def firmware_grouping(runs: pd.DataFrame) -> dict:
    """Reboot-error rate by firmware version, from telemetry alone (no ground truth)."""
    rows = []
    for fw, g in runs.groupby("firmware"):
        k, n = int(g["n_reboot"].gt(0).sum()), int(len(g))
        lo, hi = _wilson(k, n)
        rows.append({"firmware": fw, "runs": n, "devices": int(g["device_id"].nunique()),
                     "runs_with_reboot": k, "reboots_per_100_runs": round(100 * k / n, 3),
                     "ci95_low_per_100": round(100 * lo, 3), "ci95_high_per_100": round(100 * hi, 3)})
    rows.sort(key=lambda r: r["firmware"])
    worst = max(rows, key=lambda r: r["reboots_per_100_runs"])
    others = [r for r in rows if r["firmware"] != worst["firmware"]]
    ok_k, ok_n = sum(r["runs_with_reboot"] for r in others), sum(r["runs"] for r in others)
    return {"by_firmware": rows, "highest_rate_firmware": worst["firmware"],
            "rate_ratio_vs_all_other_versions": round(worst["reboots_per_100_runs"] / (100 * ok_k / ok_n), 2),
            "source": "telemetry only; ground truth not used"}
