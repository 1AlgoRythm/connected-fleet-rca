"""Turn raw run rows (SQLite) into one feature row per device per day.

The big idea: every unit is compared with ITSELF. Each signal is divided by that
device's own mean over its first BASELINE_DAYS days (its "as-shipped" fingerprint),
so a heavy-carpet home with a naturally high brush current is not an outlier.
All windows end on day d and look only backwards (causal): at day d the model
never sees day d+1. Hard-failure error codes are deliberately NOT features.

Array shape convention in comments: (R,) = one value per run in the window.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from . import config as C

HARD_CODES = {"E_BRUSH_JAM", "E_SUCTION_LOW", "E_BATT_LOW", "E_WHEEL_STALL"}   # excluded from features
MIN_RUNS_7D = 3

RATIO_FEATURES = ["suction_r7", "brush_r7", "wheel_l_r7", "wheel_r_r7", "wheel_max_r7", "wheel_asym",
                  "airflow_r7", "cap_r7", "reff_r7", "bin_r7", "brush_cv7",
                  "suction_slope14", "brush_slope14", "airflow_slope14", "cap_slope14",
                  "reff_slope14", "wheel_max_slope14"]
ERROR_FEATURES = ["reboot_rate14", "benign_err_rate14", "warn_rate14"]
FIRMWARE_FEATURES = ["fw_ordinal", "days_since_fw_change"]
MODEL_FEATURES = RATIO_FEATURES + ERROR_FEATURES + FIRMWARE_FEATURES
RAW_FEATURES = ["raw_brush_a", "raw_suction_a", "raw_airflow_lps", "raw_cap_mah", "raw_sag_v",
                "raw_wheel_max_a", "reboot_count7"]

# Which feature columns belong to which physical signal (used to explain a flag).
SIGNAL_GROUPS = {
    "suction_motor_current": ["suction_r7", "suction_slope14"],
    "brush_motor_current": ["brush_r7", "brush_slope14", "brush_cv7"],
    "wheel_motor_currents": ["wheel_l_r7", "wheel_r_r7", "wheel_max_r7", "wheel_asym", "wheel_max_slope14"],
    "airflow": ["airflow_r7", "airflow_slope14"],
    "battery_capacity": ["cap_r7", "cap_slope14"],
    "battery_resistance": ["reff_r7", "reff_slope14"],
    "reboot_errors": ["reboot_rate14"],
    "other_errors": ["benign_err_rate14", "warn_rate14"],
    "bin_full_events": ["bin_r7"],
    "firmware_version": FIRMWARE_FEATURES,
}


def load_runs(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY device_id, ts", con)
    con.close()
    df["day"] = (df["ts"] - C.SIM_EPOCH) // 86400
    codes = df["error_codes"].map(json.loads)
    df["n_reboot"] = codes.map(lambda c: c.count("E_REBOOT"))
    df["n_benign"] = codes.map(lambda c: sum(x in ("E_CLIFF", "E_STUCK") for x in c))
    df["n_warn"] = codes.map(lambda c: c.count("E_BRUSH_STALL"))
    mf = df["power_mode"].map(C.POWER_MODES)
    df["suction_n"] = df["suction_a"] / mf                         # A, power-mode divided out
    df["airflow_n"] = df["airflow_lps"] / mf ** 0.6                # L/s, same idea
    motors = df["suction_a"] + df["brush_a"] + df["wheel_l_a"] + df["wheel_r_a"] + C.NOMINAL["misc_a"]
    df["reff"] = df["v_sag_at_start_v"] / motors                  # ohm-like: sag per amp drawn
    df["wheel_max_a"] = df[["wheel_l_a", "wheel_r_a"]].max(axis=1)
    return df


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y on x (per day); NaN if too few distinct points."""
    if len(x) < 5 or np.ptp(x) < 3:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def device_features(g: pd.DataFrame) -> pd.DataFrame:
    base_mask = (g["day"] < C.BASELINE_DAYS).to_numpy()
    if base_mask.sum() < 3:
        return pd.DataFrame()
    cols = {"suction": "suction_n", "brush": "brush_a", "wl": "wheel_l_a", "wr": "wheel_r_a",
            "wmax": "wheel_max_a", "air": "airflow_n", "cap": "battery_capacity_mah", "reff": "reff"}
    v = {k: g[c].to_numpy(float) for k, c in cols.items()}                       # each (N,)
    base = {k: v[k][base_mask].mean() for k in v}
    r = {k: v[k] / base[k] for k in v}                                            # ratio to own baseline
    day = g["day"].to_numpy()
    area = g["area_m2"].to_numpy(float)
    bins = g["bin_full_events"].to_numpy(float)
    base_bin = bins[base_mask].sum() / max(area[base_mask].sum(), 1e-9)
    reboot, benign, warn = (g[c].to_numpy(float) for c in ("n_reboot", "n_benign", "n_warn"))
    fw = g["firmware"].to_numpy()
    fw_ord = np.array([C.FW_ORDINAL[f] for f in fw])
    change_days = day[np.r_[False, fw[1:] != fw[:-1]]]
    rows = []
    for d in range(C.BASELINE_DAYS, int(day.max()) + 1):
        w7 = (day > d - 7) & (day <= d)
        w14 = (day > d - 14) & (day <= d)
        if w7.sum() < MIN_RUNS_7D:
            continue
        past = change_days[change_days <= d]
        row = {
            "device_id": g["device_id"].iat[0], "day": d,
            "suction_r7": r["suction"][w7].mean(), "brush_r7": r["brush"][w7].mean(),
            "wheel_l_r7": r["wl"][w7].mean(), "wheel_r_r7": r["wr"][w7].mean(),
            "wheel_max_r7": r["wmax"][w7].mean(),
            "airflow_r7": r["air"][w7].mean(), "cap_r7": r["cap"][w7].mean(),
            "reff_r7": r["reff"][w7].mean(),
            "bin_r7": (bins[w7].sum() / max(area[w7].sum(), 1e-9)) / max(base_bin, 1e-9),
            "brush_cv7": r["brush"][w7].std() / r["brush"][w7].mean(),
            "reboot_rate14": reboot[w14].mean(), "benign_err_rate14": benign[w14].mean(),
            "warn_rate14": warn[w14].mean(),
            "fw_ordinal": fw_ord[w7][-1],
            "days_since_fw_change": min(d - past.max(), 60) if len(past) else min(d, 60),
            "raw_brush_a": v["brush"][w7].mean(), "raw_suction_a": g["suction_a"].to_numpy()[w7].mean(),
            "raw_airflow_lps": g["airflow_lps"].to_numpy()[w7].mean(),
            "raw_cap_mah": v["cap"][w7].mean(), "raw_sag_v": g["v_sag_at_start_v"].to_numpy()[w7].mean(),
            "raw_wheel_max_a": v["wmax"][w7].mean(), "reboot_count7": reboot[w7].sum(),
        }
        row["wheel_asym"] = float(np.log(row["wheel_l_r7"] / row["wheel_r_r7"]))
        for name, key in [("suction", "suction"), ("brush", "brush"), ("airflow", "air"),
                          ("cap", "cap"), ("reff", "reff"), ("wheel_max", "wmax")]:
            row[f"{name}_slope14"] = _slope(day[w14].astype(float), r[key][w14])
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(runs: pd.DataFrame) -> pd.DataFrame:
    parts = [device_features(g) for _, g in runs.groupby("device_id", sort=True)]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def attach_labels(feats: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Row label = injected mode once the fault has started, else 'healthy'. Eval-only columns."""
    t = truth.set_index("device_id")
    mode = feats["device_id"].map(t["mode"])
    onset = feats["device_id"].map(t["onset_day"])
    faulty_now = mode.ne("healthy") & (feats["day"] >= onset)
    out = feats.copy()
    out["row_label"] = np.where(faulty_now, mode, "healthy")
    out["device_mode"] = mode
    return out
