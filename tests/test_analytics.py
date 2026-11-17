import json

import numpy as np
import pandas as pd

from fleet import config as C
from fleet.detectors import alert_days, calibrate_threshold
from fleet.evaluate import split_devices
from fleet.features import MODEL_FEATURES, attach_labels, build_features, load_runs


def test_split_is_by_device_and_stratified():
    truth = pd.DataFrame(json.load(open("results/ground_truth.json")))
    tr, te = split_devices(truth)
    assert not set(tr) & set(te)
    assert len(tr) + len(te) == len(truth)
    share = truth.set_index("device_id").loc[te, "mode"].value_counts(normalize=True)
    assert abs(share["healthy"] - 0.58) < 0.05


def test_alert_requires_three_consecutive_flags():
    rows = pd.DataFrame({"device_id": ["a"] * 6, "day": range(6)})
    flags = np.array([1, 1, 0, 1, 1, 1], bool)
    assert alert_days(rows, flags).iloc[0] == 5                  # third day of the run 3,4,5
    assert np.isnan(alert_days(rows, np.array([1, 1, 0, 1, 1, 0], bool)).iloc[0])


def test_threshold_respects_healthy_budget():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"device_id": np.repeat([f"d{i}" for i in range(40)], 30),
                         "day": np.tile(np.arange(30), 40)})
    scores = pd.Series(rng.normal(size=len(rows)))
    healthy = pd.Series(True, index=[f"d{i}" for i in range(40)])
    tau = calibrate_threshold(rows, scores, healthy, target=0.05)
    alerted = alert_days(rows, (scores >= tau).to_numpy()).notna().mean()
    assert alerted <= 0.05


def test_features_are_causal_and_label_free(small_fleet):
    runs = load_runs(small_fleet["db"])
    feats = build_features(runs)
    truth = pd.DataFrame(small_fleet["truth"])
    rows = attach_labels(feats, truth)
    assert set(MODEL_FEATURES) <= set(feats.columns)
    assert not {"row_label", "device_mode", "mode", "onset_day"} & set(MODEL_FEATURES)
    # causality: truncating the data after day 40 must not change features on or before day 40
    cut = runs[runs["day"] <= 40]
    feats_cut = build_features(cut)
    # (the full table may hold extra late days for a device whose data ended before day 40 in the cut)
    key = ["device_id", "day"]
    a = feats[feats["day"] <= 40].set_index(key)[MODEL_FEATURES]
    b = feats_cut.set_index(key)[MODEL_FEATURES]
    assert set(b.index) <= set(a.index)
    pd.testing.assert_frame_equal(a.loc[b.index], b)
    assert rows["row_label"].isin(C.MODES).all()
