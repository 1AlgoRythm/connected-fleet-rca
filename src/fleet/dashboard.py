"""Streamlit dashboard: fleet health, flagged devices with predicted mode + top signals, per-device plot.

Run:  uv run streamlit run src/fleet/dashboard.py
Needs data/fleet.db and results/*.json from the pipeline. Verdicts shown are the
OUT-OF-FOLD ones (each device judged by a model that never trained on it).
"""
from __future__ import annotations

import json
import os
import sqlite3

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fleet import config as C  # noqa: E402

DB = os.environ.get("FLEET_DB", "data/fleet.db")
RES = os.environ.get("FLEET_RESULTS", "results")


@st.cache_data
def load():
    con = sqlite3.connect(DB)
    runs = pd.read_sql_query("SELECT * FROM runs ORDER BY device_id, ts", con)
    con.close()
    runs["day"] = (runs["ts"] - C.SIM_EPOCH) / 86400.0
    dev = pd.DataFrame(json.load(open(f"{RES}/dashboard_devices.json")))
    par = json.load(open(f"{RES}/pareto.json"))
    fw = json.load(open(f"{RES}/firmware_grouping.json"))
    return runs, dev, par, fw


st.set_page_config(page_title="Fleet RCA (simulated)", layout="wide")
st.title("Robot-vacuum fleet health  |  SIMULATED data, no real devices")
runs, dev, par, fw = load()
flagged = dev[dev["predicted_mode"] != "healthy"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Devices", len(dev))
c2.metric("Flagged", len(flagged))
c3.metric("Flagged share", f"{100 * len(flagged) / len(dev):.1f} %")
c4.metric("Runs stored", f"{len(runs):,}")

left, right = st.columns(2)
with left:
    st.subheader("Failure-mode Pareto (flagged devices)")
    p = pd.DataFrame(par["predicted"]).set_index("mode")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(p.index, p["devices"], color="#3b6ea5")
    ax2 = ax.twinx()
    ax2.plot(p.index, 100 * p["cumulative_share"], color="#d1495b", marker="o")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("cumulative %")
    ax.set_ylabel("devices")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)
with right:
    st.subheader("Reboot errors by firmware version")
    f = pd.DataFrame(fw["by_firmware"]).set_index("firmware")
    st.bar_chart(f["reboots_per_100_runs"])
    st.caption(f"Highest: {fw['highest_rate_firmware']}, "
               f"{fw['rate_ratio_vs_all_other_versions']}x the rate of all other versions (telemetry only).")

st.subheader("Flagged devices")
table = flagged.assign(top_signals=flagged["top_signals"].map(lambda s: ", ".join(x["signal"] for x in s)))
st.dataframe(table[["device_id", "predicted_mode", "alert_day", "top_signals",
                    "injected_mode_ground_truth"]].sort_values("alert_day"), height=240, use_container_width=True)

st.subheader("Telemetry for one device")
choice = st.selectbox("Device", list(flagged["device_id"]) + list(dev[dev["predicted_mode"] == "healthy"]["device_id"][:20]))
d = runs[runs["device_id"] == choice]
row = dev[dev["device_id"] == choice].iloc[0]
fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
for ax, (col, label) in zip(axes.ravel(), [("brush_a", "brush motor (A)"), ("suction_a", "suction motor (A)"),
                                           ("wheel_l_a", "left wheel (A)"), ("airflow_lps", "airflow (L/s)"),
                                           ("battery_capacity_mah", "capacity (mAh)"),
                                           ("v_sag_at_start_v", "voltage sag at start (V)")]):
    ax.plot(d["day"], d[col], ".", ms=3)
    if col == "wheel_l_a":
        ax.plot(d["day"], d["wheel_r_a"], ".", ms=3, color="#d1495b")
    if row["alert_day"] is not None and not pd.isna(row["alert_day"]):
        ax.axvline(row["alert_day"], color="red", ls="--", lw=1)
    ax.set_title(label, fontsize=9)
for ax in axes[1]:
    ax.set_xlabel("day")
st.pyplot(fig)
st.caption(f"{choice}: predicted '{row['predicted_mode']}'; red line = first alert day. "
           f"Injected truth (simulation only): {row['injected_mode_ground_truth']}.")
