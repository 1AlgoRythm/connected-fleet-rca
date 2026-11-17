"""Step 1 of the pipeline: simulate the fleet, push it through MQTT, land it in SQLite.

Writes data/fleet.db (ignored by git), results/ground_truth.json and results/ingest_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

from . import config as C
from .broker import BrokerProcess, DEFAULT_PORT
from .ingest import Ingest
from .publisher import publish_all
from .simulate import simulate_fleet


def run(db_path: str, results_dir: str, port: int = DEFAULT_PORT, n_devices: int = C.N_DEVICES,
        horizon: int = C.HORIZON_DAYS, n_bad: int = 20, write_truth: bool = True,
        puback_timeout: float | None = None, report_name: str = "ingest_report.json",
        wait_timeout: float = 300, max_gapfill_rounds: int = 3) -> dict:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    t = time.time()
    msgs, truth = simulate_fleet(C.SEED, n_devices, horizon)
    sim_seconds = time.time() - t
    if write_truth:
        with open(os.path.join(results_dir, "ground_truth.json"), "w") as f:
            json.dump(truth, f, indent=1)

    def stored_keys() -> set:
        con = sqlite3.connect(db_path)
        keys = set(con.execute("SELECT device_id, run_id FROM runs"))
        con.close()
        return keys

    rounds = []
    with BrokerProcess(port, puback_timeout):
        ing = Ingest(db_path, port).start()   # returns only after the SUBACK
        t0 = time.time()
        pub = publish_all(msgs, port, n_clients=4, n_bad=n_bad)
        ok = ing.wait_until(pub["messages_published"], timeout=wait_timeout)
        t1 = ing.last_ts or time.time()          # moment the last first-pass message hit SQLite
        first_pass = {"received": ing.stats["received"], "rejected": ing.stats["rejected"]}
        # Gap-fill: every device numbers its runs 0,1,2,... and keeps them in a local buffer, so the
        # cloud can ask for exactly the sequence numbers it is missing. The broker's ack to the
        # publisher says nothing about the broker-to-subscriber leg, so this check is necessary.
        have = stored_keys()
        first_pass["stored"] = len(have)
        for rnd in range(max_gapfill_rounds):
            missing = [m for m in msgs if (m["device_id"], m["run_id"]) not in have]
            if not missing:
                break
            before = ing.stats["received"]
            pr = publish_all(missing, port, n_clients=2, n_bad=0)
            ing.wait_until(before + pr["messages_published"], timeout=60)
            have = stored_keys()
            rounds.append({"round": rnd + 1, "republished": pr["messages_published"], "stored_after": len(have)})
        ing.stop()

    con = sqlite3.connect(db_path)
    stored_rows = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    distinct = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT device_id, run_id FROM runs)").fetchone()[0]
    con.close()
    wall = t1 - t0
    report = {
        "n_devices": n_devices, "horizon_days": horizon, "seed": C.SEED,
        "simulation_seconds": round(sim_seconds, 2),
        "messages_generated": len(msgs),
        **pub,
        "ingest_received": ing.stats["received"],
        "ingest_rejected": ing.stats["rejected"],
        "reject_reasons": ing.reject_reasons,
        "ingest_duplicates_ignored": ing.stats["duplicates"],
        "rows_stored": stored_rows,
        "distinct_rows_stored": distinct,
        "first_pass": {**first_pass,
                       "valid_lost_in_transit": pub["valid_payloads"] - (first_pass["stored"]),
                       "loss_rate": round((pub["valid_payloads"] - first_pass["stored"]) / pub["valid_payloads"], 6)},
        "gapfill_rounds": rounds,
        "final_missing_after_gapfill": len(msgs) - distinct,
        "first_pass_drained": ok,
        "first_pass_end_to_end_seconds": round(wall, 3),
        "first_pass_throughput_msgs_per_s": round(pub["messages_published"] / wall, 1),
        "broker": "amqtt (pure Python, in-process child) on 127.0.0.1",
        "qos": 1,
        "broker_qos1_puback_timeout_s": puback_timeout,   # None = wait forever
    }
    with open(os.path.join(results_dir, report_name), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/fleet.db")
    ap.add_argument("--results", default="results")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--default-puback-timeout", action="store_true",
                    help="negative control: broker with amqtt's stock 5 s QoS1 ack timeout")
    a = ap.parse_args()
    if a.default_puback_timeout:
        out = run("data/fleet_default_timeout.db", a.results, a.port, write_truth=False,
                  puback_timeout=5.0, report_name="ingest_report_default_timeout.json", wait_timeout=60,
                  max_gapfill_rounds=0)
    else:
        out = run(a.db, a.results, a.port)
    print(json.dumps(out, indent=1))
