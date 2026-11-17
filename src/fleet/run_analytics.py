"""Step 2 of the pipeline: features -> detectors -> metrics, RCA, checks -> results/*.json."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .evaluate import fit_and_judge, score_detector, split_devices
from .explain import (explain_flagged, explanation_hit_rate, firmware_grouping, pareto,
                      permutation_importance_groups)
from .features import (ERROR_FEATURES, FIRMWARE_FEATURES, MODEL_FEATURES, RATIO_FEATURES,
                       attach_labels, build_features, load_runs)

GROUND_TRUTH_COLUMNS = {"mode", "onset_day", "hard_failure_day", "severity_scale", "row_label", "device_mode"}


def dump(obj, results_dir: str, name: str) -> None:
    with open(os.path.join(results_dir, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def main(db: str, results_dir: str) -> None:
    truth = pd.DataFrame(json.load(open(os.path.join(results_dir, "ground_truth.json"))))
    runs = load_runs(db)
    rows = attach_labels(build_features(runs), truth)
    assert not GROUND_TRUTH_COLUMNS & set(MODEL_FEATURES), "a label column is used as a feature"
    train_ids, test_ids = split_devices(truth, 0.4, seed=0)
    tmode = truth.set_index("device_id")["mode"]

    # ---------------------------------------------------------------- main split
    res = fit_and_judge(rows, truth, train_ids, test_ids)
    metrics = {
        "protocol": {
            "split": "by device, stratified by injected mode, 60/40, seed 0; no device in both sets",
            "train_devices": len(train_ids), "test_devices": len(test_ids),
            "train_modes": tmode.loc[train_ids].value_counts().to_dict(),
            "test_modes": tmode.loc[test_ids].value_counts().to_dict(),
            "feature_rows": int(len(rows)), "debounce_days": 3,
            "threshold_rule": "per detector, smallest threshold with <=5% of healthy TRAIN devices alerting "
                              "(GB: on out-of-fold scores); then frozen for the test devices",
            "verdict_rule": "mode named from all rows from the first alert day to the end of the device's data",
            "hard_failure_codes_in_features": False},
        "thresholds": {k: res[k] for k in ("rule_tau", "iforest_tau", "gb_tau")},
        "rule_baseline": score_detector(res["rule"], truth, "rule"),
        "isolation_forest_unsupervised": score_detector(res["iforest"], truth, "iforest"),
        "gradient_boosting": score_detector(res["gb"], truth, "gb"),
    }
    dump(metrics, results_dir, "metrics.json")

    expl = explain_flagged(res)
    dump({"explanations_test_devices": expl, "hit_rate": explanation_hit_rate(expl, truth),
          "permutation_importance": permutation_importance_groups(res)}, results_dir, "rca_test_devices.json")
    dump(firmware_grouping(runs), results_dir, "firmware_grouping.json")

    # ----------------------------------------------- 5-fold device CV (fleet-wide, out-of-fold)
    ids = truth["device_id"].to_numpy()
    folds, oof_gb, oof_expl, cv_rows = [], [], [], []
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, truth["mode"])):
        r = fit_and_judge(rows, truth, list(ids[a]), list(ids[b]), seed=k)
        row = {"fold": k, "test_devices": len(b)}
        for name, key in [("rule", "rule"), ("iforest", "iforest"), ("gb", "gb")]:
            m = score_detector(r[key], truth, name)
            row[name] = {"macro_f1_fault_modes": m.get("macro_f1_fault_modes"),
                         "any_fault_recall": m["detection"]["any_fault_recall"],
                         "healthy_false_alarm_rate": m["detection"]["healthy_false_alarm_rate"],
                         "lead_median": m["lead_time"]["all_degrading"]["lead_days_median"]}
        folds.append(row)
        oof_gb.append(r["gb"])
        oof_expl += explain_flagged(r)
    allv = pd.concat(oof_gb, ignore_index=True)
    summary = {}
    for name in ("rule", "iforest", "gb"):
        for metric in folds[0][name]:
            vals = [f[name][metric] for f in folds if f[name][metric] is not None]
            if vals:
                summary.setdefault(name, {})[metric] = {"mean": round(float(np.mean(vals)), 4),
                                                        "std": round(float(np.std(vals)), 4), "n_folds": len(vals)}
    dump({"folds": folds, "summary": summary}, results_dir, "cv_summary.json")
    dump(pareto(allv, truth), results_dir, "pareto.json")
    top = {e["device_id"]: e for e in oof_expl}
    dash = []
    for _, v in allv.iterrows():
        e = top.get(v["device_id"])
        dash.append({"device_id": v["device_id"], "predicted_mode": v["verdict"],
                     "alert_day": None if np.isnan(v["alert_day"]) else int(v["alert_day"]),
                     "top_signals": e["top_signals"] if e else [],
                     "injected_mode_ground_truth": tmode[v["device_id"]]})
    dump(dash, results_dir, "dashboard_devices.json")

    # ----------------------------------------------------------- leakage / easiness checks
    checks: dict = {"split_overlap_devices": len(set(train_ids) & set(test_ids)),
                    "ground_truth_columns_in_model_features": sorted(GROUND_TRUTH_COLUMNS & set(MODEL_FEATURES))}
    variants = {
        "shuffled_training_labels": dict(shuffle_labels=True),
        "no_firmware_features": dict(gb_features=RATIO_FEATURES + ERROR_FEATURES),
        "no_error_or_firmware_features": dict(gb_features=RATIO_FEATURES),
        "no_slope_features": dict(gb_features=[f for f in MODEL_FEATURES if "slope" not in f]),
    }
    for name, kw in variants.items():
        r = fit_and_judge(rows, truth, train_ids, test_ids, gb_only=True, **kw)
        m = score_detector(r["gb"], truth, "gb")
        checks[name] = {"macro_f1_fault_modes": m["macro_f1_fault_modes"],
                        "any_fault_recall": m["detection"]["any_fault_recall"],
                        "healthy_false_alarm_rate": m["detection"]["healthy_false_alarm_rate"],
                        "recall_by_mode": m["detection"]["recall_by_mode"]}
    v = res["gb"].copy()
    v["scale"] = v["device_id"].map(truth.set_index("device_id")["severity_scale"])
    v["true"] = v["device_id"].map(tmode)
    f = v[v["true"] != "healthy"].copy()
    f["bin"] = pd.cut(f["scale"], [0, 0.6, 0.9, 1.3], labels=["mild(0.3-0.6)", "medium(0.6-0.9)", "strong(0.9-1.2)"])
    checks["gb_by_fault_strength"] = {
        str(b): {"devices": int(len(g)), "alerted_share": round(float(g["alert_day"].notna().mean()), 4),
                 "correct_mode_share": round(float((g["verdict"] == g["true"]).mean()), 4)}
        for b, g in f.groupby("bin", observed=True)}
    dump(checks, results_dir, "leakage_checks.json")
    print("analytics done ->", results_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/fleet.db")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    main(a.db, a.results)
