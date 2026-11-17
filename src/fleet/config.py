"""Shared constants. Every physical quantity carries its unit in the name or comment."""
from __future__ import annotations

SEED = 20261116
N_DEVICES = 600
HORIZON_DAYS = 90            # simulated days of field life per device
BASELINE_DAYS = 10           # first days used as each device's own "as-shipped" baseline
FAULT_ONSET_RANGE = (14, 50)     # day the fault starts (after the baseline window)
FAULT_DURATION_RANGE = (20, 40)  # days from onset to hard failure (severity 0 -> 1)
SIM_EPOCH = 1_767_571_200    # 2026-01-05 00:00:00 UTC, a purely fictional start date

MODES = ["healthy", "brush_hair_wrap", "clogged_filter", "battery_degradation",
         "wheel_wear", "firmware_regression"]
FAULT_MODES = MODES[1:]
DEGRADING_MODES = ["brush_hair_wrap", "clogged_filter", "battery_degradation", "wheel_wear"]
# Fractions of the fleet given each injected fault (the rest are healthy).
MODE_FRACTION = {"brush_hair_wrap": 0.08, "clogged_filter": 0.08, "battery_degradation": 0.08,
                 "wheel_wear": 0.08}
N_FW_REGRESSION = 60         # devices that get the bad OTA AND are susceptible
N_FW_BENIGN_OTA = 40         # devices that get the same OTA but are NOT susceptible

FW_ORDINAL = {"2.2.4": 0, "2.3.0": 1, "2.3.1": 2}   # 2.3.1 is the regression build
BAD_FW = "2.3.1"

# Cleaning power modes: the controller commands a setpoint, so analytics may divide it out.
POWER_MODES = {"eco": 0.75, "auto": 1.00, "max": 1.30}   # relative fan speed (unitless)

# Nominal electrical values of a healthy, new robot (4-cell Li-ion pack, ~14.4 V nominal).
NOMINAL = {
    "suction_a": 2.10,        # A, main fan motor, auto mode
    "brush_a": 0.35,          # A, brush-roll motor
    "wheel_a": 0.22,          # A, each drive wheel motor
    "airflow_lps": 9.0,       # L/s through the fan at the inlet
    "capacity_mah": 5200.0,   # mAh new pack
    "resistance_ohm": 0.14,   # ohm pack internal resistance (new)
    "v_full": 16.55,          # V at full charge (4 x 4.14 V)
    "misc_a": 0.45,           # A electronics, lidar, wifi
}
