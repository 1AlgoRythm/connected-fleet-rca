"""Telemetry schema: the ingest service trusts nothing it reads off the wire.

validate() returns (message, None) when OK or (None, reason) when rejected.
Ranges are physical sanity limits (a pack cannot read 40 V), not statistical outliers.
"""
from __future__ import annotations

import json
import math

# field -> (type, min, max). Units are in the field names / docs in simulate.py.
NUMERIC = {
    "runtime_min": (0, 300), "area_m2": (0, 400),
    "battery_v_start": (8, 20), "battery_v_min": (4, 20), "v_sag_at_start_v": (-1, 10),
    "battery_capacity_mah": (500, 8000),
    "suction_a": (0, 10), "brush_a": (0, 10), "wheel_l_a": (0, 10), "wheel_r_a": (0, 10),
    "airflow_lps": (0, 40),
}
INTEGER = {"run_id": (0, 10**9), "ts": (1_500_000_000, 2_500_000_000),
           "bin_full_events": (0, 1000), "schema_version": (1, 1)}
STRING = ("device_id", "firmware", "power_mode")
REQUIRED = set(NUMERIC) | set(INTEGER) | set(STRING) | {"error_codes"}


def validate(topic: str, payload: bytes) -> tuple[dict | None, str | None]:
    try:
        msg = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None, "not_json"
    if not isinstance(msg, dict):
        return None, "not_object"
    missing = REQUIRED - msg.keys()
    if missing:
        return None, f"missing:{sorted(missing)[0]}"
    for k in STRING:
        if not isinstance(msg[k], str) or not msg[k]:
            return None, f"bad_type:{k}"
    for k, (lo, hi) in INTEGER.items():
        v = msg[k]
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            return None, f"bad_value:{k}"
    for k, (lo, hi) in NUMERIC.items():
        v = msg[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or not lo <= v <= hi:
            return None, f"bad_value:{k}"
    codes = msg["error_codes"]
    if not isinstance(codes, list) or not all(isinstance(c, str) and c for c in codes):
        return None, "bad_type:error_codes"
    parts = topic.split("/")
    if len(parts) != 3 or parts[0] != "fleet" or parts[2] != "telemetry" or parts[1] != msg["device_id"]:
        return None, "topic_device_mismatch"   # a device may only speak for itself
    return msg, None
