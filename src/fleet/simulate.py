"""Fleet simulator: turns a seed into per-run telemetry plus ground-truth fault labels.

Reading order: make_fleet() decides who is healthy / faulty and draws each device's
static traits; simulate_device() walks day by day and produces run messages.

Design rule: a fault multiplies a signal by (1 + effect * severity) where severity
climbs 0 -> 1 between `onset_day` and `hard_failure_day`. Healthy devices ALSO move
(carpet, pet hair, filter dirt that gets washed, slow battery fade) so a naive
"value went up" rule produces false alarms.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def _lognormal(rng: np.random.Generator, sigma: float) -> float:
    return float(np.exp(rng.normal(0.0, sigma)))


def make_fleet(seed: int = C.SEED, n_devices: int = C.N_DEVICES) -> list[dict]:
    """Return one dict of static traits + ground truth per device."""
    root = np.random.SeedSequence(seed)
    plan_rng = np.random.default_rng(root.spawn(1)[0])
    child_seeds = root.spawn(n_devices)

    # Assign each device exactly one role so class counts are exact and reproducible.
    roles: list[tuple[str, bool]] = []  # (mode, gets_bad_ota)
    for mode, frac in C.MODE_FRACTION.items():
        roles += [(mode, False)] * round(frac * n_devices)
    scale = n_devices / C.N_DEVICES   # small fleets (tests) keep the same proportions
    roles += [("firmware_regression", True)] * round(C.N_FW_REGRESSION * scale)
    roles += [("healthy", True)] * round(C.N_FW_BENIGN_OTA * scale)   # same OTA, not susceptible
    roles += [("healthy", False)] * (n_devices - len(roles))
    plan_rng.shuffle(roles)

    fleet = []
    for i, (mode, bad_ota) in enumerate(roles):
        rng = np.random.default_rng(child_seeds[i])
        dev = {
            "device_id": f"SV-{i:04d}",
            "seed_index": i,
            "mode": mode,
            "bad_ota": bad_ota,
            # ---- static traits (the "realistic variation" of healthy units) ----
            "carpet_frac": float(rng.beta(2, 3)),            # share of runs on carpet
            "pets": bool(rng.random() < 0.4),                 # pet homes: faster hair build-up
            "suction_a": C.NOMINAL["suction_a"] * _lognormal(rng, 0.06),
            "brush_a": C.NOMINAL["brush_a"] * _lognormal(rng, 0.08),
            "wheel_l_a": C.NOMINAL["wheel_a"] * _lognormal(rng, 0.08),
            "wheel_r_a": C.NOMINAL["wheel_a"] * _lognormal(rng, 0.08),
            "airflow_lps": C.NOMINAL["airflow_lps"] * _lognormal(rng, 0.05),
            "capacity_mah": C.NOMINAL["capacity_mah"] * float(rng.normal(1.0, 0.02)),
            "resistance_ohm": C.NOMINAL["resistance_ohm"] * _lognormal(rng, 0.10),
            "healthy_fade_per_day": 0.0005 * _lognormal(rng, 0.5),   # fraction/day, heavy-tailed
            # healthy confounders: some homes shed far more hair / dirt than average
            "hair_gain": _lognormal(rng, 0.5), "dirt_gain": _lognormal(rng, 0.4),
            # a third of homes change floor mix mid-life (new rug, moved to another room)
            "shift_day": int(rng.integers(15, 70)) if rng.random() < 0.33 else None,
            "carpet_frac_after": float(rng.beta(2, 3)),
            "filter_wash_days": int(rng.integers(14, 36)),    # owner washes filter this often
            "hair_clear_days": int(rng.integers(10, 22)),     # owner clears brush this often
            "start_fw": "2.2.4" if rng.random() < 0.6 else "2.3.0",
            "ota_day": int(rng.integers(15, 41)),            # day the 2.3.1 OTA lands (if any)
            "benign_ota_day": int(rng.integers(10, 60)),     # unrelated 2.2.4 -> 2.3.0 OTA
            "wheel_side": "l" if rng.random() < 0.5 else "r",
            "severity_scale": float(rng.uniform(0.3, 1.2)),  # mild ... strong fault
            "onset_day": None,
            "hard_failure_day": None,
        }
        if mode in C.DEGRADING_MODES:
            dev["onset_day"] = int(rng.integers(*C.FAULT_ONSET_RANGE))
            duration = int(rng.integers(*C.FAULT_DURATION_RANGE))
            dev["duration_days"] = duration
            dev["hard_failure_day"] = dev["onset_day"] + duration
        if mode == "firmware_regression":
            dev["onset_day"] = dev["ota_day"]  # reboots start the day the bad build lands
        fleet.append(dev)
    return fleet


def firmware_on_day(dev: dict, day: int) -> str:
    if dev["bad_ota"] and day >= dev["ota_day"]:
        return C.BAD_FW
    if dev["start_fw"] == "2.2.4" and not dev["bad_ota"] and day >= dev["benign_ota_day"]:
        return "2.3.0"
    return dev["start_fw"]


def severity(dev: dict, mode: str, day: int) -> float:
    """0 before onset, 1 at the hard-failure day, linear in between."""
    if dev["mode"] != mode or dev["onset_day"] is None or dev["hard_failure_day"] is None:
        return 0.0
    return float(np.clip((day - dev["onset_day"]) / dev["duration_days"], 0.0, 1.0))


def simulate_device(dev: dict, seed: int = C.SEED, horizon: int = C.HORIZON_DAYS) -> list[dict]:
    """Walk one device through its life and return its run messages (dicts)."""
    rng = np.random.default_rng([seed, dev["seed_index"], 7])
    m = dev["severity_scale"]
    messages: list[dict] = []
    run_id = 0
    for day in range(horizon):
        s_brush = severity(dev, "brush_hair_wrap", day)
        s_clog = severity(dev, "clogged_filter", day)
        s_batt = severity(dev, "battery_degradation", day)
        s_wheel = severity(dev, "wheel_wear", day)
        failed = dev["hard_failure_day"] is not None and day >= dev["hard_failure_day"]

        # --- slow state of the healthy world --------------------------------
        # hair builds up to ~+25 % brush current in pet homes, then the owner clears it
        hair = (0.25 if dev["pets"] else 0.08) * dev["hair_gain"] * ((day % dev["hair_clear_days"]) / dev["hair_clear_days"])
        # filter dirt: up to ~12 % airflow loss, then the owner washes it
        dirt = 0.12 * dev["dirt_gain"] * ((day % dev["filter_wash_days"]) / dev["filter_wash_days"])
        carpet_frac = (dev["carpet_frac_after"] if dev["shift_day"] is not None and day >= dev["shift_day"]
                       else dev["carpet_frac"])
        soh = 1.0 - dev["healthy_fade_per_day"] * day - 0.20 * m * s_batt   # state of health
        # pack resistance grows a little with ordinary ageing and a lot with the fault
        resist = dev["resistance_ohm"] * (1.0 + 2.0 * (1.0 - soh) + 1.6 * m * s_batt)   # ohm
        fw = firmware_on_day(dev, day)

        n_runs = int(rng.choice([0, 1, 2], p=[0.22, 0.68, 0.10]))
        for _ in range(n_runs):
            mode_name = str(rng.choice(list(C.POWER_MODES), p=[0.15, 0.70, 0.15]))
            mf = C.POWER_MODES[mode_name]
            carpet = rng.random() < carpet_frac
            noise = lambda sd: float(rng.normal(1.0, sd))  # noqa: E731  multiplicative noise

            brush_mult = (1 + hair) * (1 + 1.4 * m * s_brush ** 1.4)
            clog = dirt + 0.45 * m * s_clog ** 1.1                       # fraction of airflow lost
            suction_a = dev["suction_a"] * mf * (1 - 0.30 * clog) * (1.05 if carpet else 1.0) * noise(0.03)
            brush_a = dev["brush_a"] * brush_mult * (1.25 if carpet else 1.0) * noise(0.04)
            wheel_l = dev["wheel_l_a"] * (1.2 if carpet else 1.0) * noise(0.04)
            wheel_r = dev["wheel_r_a"] * (1.2 if carpet else 1.0) * noise(0.04)
            wear = 1 + 0.9 * m * s_wheel ** 1.2
            if dev["wheel_side"] == "l":
                wheel_l *= wear
            else:
                wheel_r *= wear
            airflow = dev["airflow_lps"] * mf ** 0.6 * (1 - clog) * (0.95 if carpet else 1.0) * noise(0.03)

            i_avg = C.NOMINAL["misc_a"] + suction_a + brush_a + wheel_l + wheel_r      # A
            usable_mah = soh * dev["capacity_mah"] * 0.85
            max_runtime_min = usable_mah / (i_avg * 1000.0) * 60.0
            job_min = float(np.clip(rng.lognormal(np.log(50), 0.35), 10, 120))
            runtime_min = min(job_min, max_runtime_min)
            area_m2 = runtime_min * 0.65 * (0.85 if carpet else 1.0) * noise(0.05)  # 0.65 m2/min
            used_frac = runtime_min * i_avg * 1000.0 / 60.0 / (soh * dev["capacity_mah"])
            v_start = float(rng.normal(C.NOMINAL["v_full"], 0.03))                  # V
            sag_v = 1.35 * i_avg * resist                                            # V at spin-up
            v_min = v_start - sag_v - 1.8 * used_frac                                # V end of run

            bin_rate = area_m2 / 45.0 * (1.3 if dev["pets"] else 1.0) * (1 + 0.6 * clog)
            errors: list[str] = []
            if rng.random() < 0.040:
                errors.append("E_CLIFF")          # cliff sensor false trigger
            if rng.random() < (0.05 if carpet else 0.02):
                errors.append("E_STUCK")          # wedged on a rug fringe
            if rng.random() < 0.003:
                errors.append("E_REBOOT")         # background reboot rate of any firmware
            if fw == C.BAD_FW and dev["mode"] == "firmware_regression" and rng.random() < 0.12:
                errors.append("E_REBOOT")         # the regression: only on one build
            if s_brush > 0.6 and rng.random() < 0.15 * s_brush:
                errors.append("E_BRUSH_STALL")    # warning before the jam
            if failed:                            # hard-failure signature, every run
                errors.append({"brush_hair_wrap": "E_BRUSH_JAM", "clogged_filter": "E_SUCTION_LOW",
                               "battery_degradation": "E_BATT_LOW", "wheel_wear": "E_WHEEL_STALL"
                               }.get(dev["mode"], ""))

            ts = C.SIM_EPOCH + day * 86400 + int(rng.integers(8 * 3600, 20 * 3600))
            messages.append({
                "schema_version": 1,
                "device_id": dev["device_id"],
                "run_id": run_id,
                "ts": ts,                                                    # s since epoch
                "firmware": fw,
                "power_mode": mode_name,
                "runtime_min": round(runtime_min, 2),                        # min
                "area_m2": round(area_m2, 2),                                # m^2
                "bin_full_events": int(rng.poisson(bin_rate)),               # count
                "error_codes": errors,
                "battery_v_start": round(v_start, 3),                        # V
                "battery_v_min": round(v_min, 3),                            # V
                "v_sag_at_start_v": round(sag_v + float(rng.normal(0, 0.03)), 3),  # V
                "battery_capacity_mah": round(soh * dev["capacity_mah"] * noise(0.015), 1),  # mAh
                "suction_a": round(suction_a, 4),                            # A
                "brush_a": round(brush_a, 4),                                # A
                "wheel_l_a": round(wheel_l, 4),                              # A
                "wheel_r_a": round(wheel_r, 4),                              # A
                "airflow_lps": round(airflow, 3),                            # L/s
            })
            run_id += 1
        if failed:
            break  # the unit is returned/dead after its hard-failure day
    return messages


def simulate_fleet(seed: int = C.SEED, n_devices: int = C.N_DEVICES,
                   horizon: int = C.HORIZON_DAYS) -> tuple[list[dict], list[dict]]:
    """Return (messages sorted by time, ground-truth rows)."""
    fleet = make_fleet(seed, n_devices)
    msgs: list[dict] = []
    for dev in fleet:
        msgs += simulate_device(dev, seed, horizon)
    msgs.sort(key=lambda r: (r["ts"], r["device_id"]))
    truth = [{k: dev[k] for k in ("device_id", "mode", "onset_day", "hard_failure_day",
                                  "severity_scale", "bad_ota", "pets", "carpet_frac")}
             for dev in fleet]
    return msgs, truth
