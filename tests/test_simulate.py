import numpy as np
import pandas as pd

from fleet import config as C
from fleet.simulate import make_fleet, simulate_fleet


def test_seeded_and_reproducible():
    a, _ = simulate_fleet(n_devices=30, horizon=30)
    b, _ = simulate_fleet(n_devices=30, horizon=30)
    assert a == b
    c, _ = simulate_fleet(seed=1, n_devices=30, horizon=30)
    assert a != c


def test_fleet_composition_and_ground_truth():
    fleet = make_fleet()
    assert len(fleet) >= 300
    modes = pd.Series([d["mode"] for d in fleet]).value_counts()
    assert modes["healthy"] > modes[C.FAULT_MODES].max()          # faults are a minority per mode
    assert (modes[C.FAULT_MODES].sum()) < len(fleet) / 2            # and overall
    for d in fleet:
        if d["mode"] in C.DEGRADING_MODES:
            assert d["hard_failure_day"] > d["onset_day"] >= C.FAULT_ONSET_RANGE[0]
        if d["mode"] == "healthy":
            assert d["onset_day"] is None


def test_faults_move_the_right_signal():
    msgs, truth = simulate_fleet()
    df = pd.DataFrame(msgs)
    df["day"] = (df["ts"] - C.SIM_EPOCH) // 86400
    t = pd.DataFrame(truth).set_index("device_id")
    df = df.join(t[["mode", "onset_day"]], on="device_id")

    def ratio(col, mode):
        d = df[df["mode"] == mode]
        late = d[d["day"] >= d["onset_day"] + 15][col].mean()
        early = d[d["day"] < 10][col].mean()
        return late / early

    assert ratio("brush_a", "brush_hair_wrap") > 1.2
    assert ratio("airflow_lps", "clogged_filter") < 0.9
    assert ratio("battery_capacity_mah", "battery_degradation") < 0.95
    assert max(ratio("wheel_l_a", "wheel_wear"), ratio("wheel_r_a", "wheel_wear")) > 1.1
    reboots = df.assign(r=df["error_codes"].map(lambda c: "E_REBOOT" in c))
    bad = reboots[(reboots["mode"] == "firmware_regression") & (reboots["firmware"] == C.BAD_FW)]["r"].mean()
    ok = reboots[reboots["mode"] == "healthy"]["r"].mean()
    assert bad > 10 * ok


def test_units_are_physical():
    msgs, _ = simulate_fleet(n_devices=60, horizon=40)
    df = pd.DataFrame(msgs)
    assert df["battery_v_min"].between(10, 17).all()
    assert df["suction_a"].between(0.5, 5).all()
    assert (df["runtime_min"] > 0).all() and (df["area_m2"] > 0).all()
    assert np.isfinite(df.select_dtypes("number")).all().all()
