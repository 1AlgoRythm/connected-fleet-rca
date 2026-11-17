"""Ingest service: MQTT subscriber -> schema validation -> SQLite.

Why a queue + writer thread: paho's network thread must return quickly or QoS 1
acks stall. The callback only enqueues; one writer thread batches inserts.
Why INSERT OR IGNORE on (device_id, run_id): QoS 1 is "at least once", so the
broker may redeliver; the primary key makes redelivery harmless and countable.
"""
from __future__ import annotations

import argparse
import json
import queue
import sqlite3
import threading
import time

import paho.mqtt.client as mqtt

from .broker import DEFAULT_PORT
from .schema import validate

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  device_id TEXT NOT NULL, run_id INTEGER NOT NULL, ts INTEGER NOT NULL, firmware TEXT,
  power_mode TEXT, runtime_min REAL, area_m2 REAL, bin_full_events INTEGER, error_codes TEXT,
  battery_v_start REAL, battery_v_min REAL, v_sag_at_start_v REAL, battery_capacity_mah REAL,
  suction_a REAL, brush_a REAL, wheel_l_a REAL, wheel_r_a REAL, airflow_lps REAL,
  PRIMARY KEY (device_id, run_id));
CREATE TABLE IF NOT EXISTS rejected (topic TEXT, reason TEXT, payload_head TEXT);
"""
COLS = ["device_id", "run_id", "ts", "firmware", "power_mode", "runtime_min", "area_m2",
        "bin_full_events", "error_codes", "battery_v_start", "battery_v_min", "v_sag_at_start_v",
        "battery_capacity_mah", "suction_a", "brush_a", "wheel_l_a", "wheel_r_a", "airflow_lps"]
INSERT = f"INSERT OR IGNORE INTO runs ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})"


class Ingest:
    def __init__(self, db_path: str, port: int = DEFAULT_PORT, client_id: str = "fleet-ingest"):
        self.db_path, self.port = db_path, port
        self.q: queue.Queue = queue.Queue()
        self.stats = {"received": 0, "stored": 0, "duplicates": 0, "rejected": 0}
        self.reject_reasons: dict[str, int] = {}
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        # clean_session=False: if we are briefly down the broker queues QoS 1 messages for us.
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                                  clean_session=False)
        self.client.on_message = lambda c, u, m: self.q.put((m.topic, bytes(m.payload)))
        self.client.on_connect = lambda c, u, f, rc, p=None: c.subscribe("fleet/+/telemetry", qos=1)
        # QoS 1 messages published while nobody is subscribed are dropped by the broker, so
        # start() must not return before the SUBACK arrives (this race lost messages under load).
        self.subscribed = threading.Event()
        self.client.on_subscribe = lambda c, u, mid, rcs, p=None: self.subscribed.set()
        self.first_ts: float | None = None
        self.last_ts: float | None = None

    def start(self) -> "Ingest":
        con = sqlite3.connect(self.db_path)
        con.executescript(SCHEMA)
        con.close()
        self._writer.start()
        self.client.connect("127.0.0.1", self.port)
        self.client.loop_start()
        if not self.subscribed.wait(timeout=15):
            raise TimeoutError("broker never acknowledged the subscription")
        return self

    def _write_loop(self) -> None:
        con = sqlite3.connect(self.db_path)
        while not (self._stop.is_set() and self.q.empty()):
            batch = []
            try:
                batch.append(self.q.get(timeout=0.2))
                while len(batch) < 1000:
                    batch.append(self.q.get_nowait())
            except queue.Empty:
                pass
            if not batch:
                continue
            now = time.time()
            self.first_ts = self.first_ts or now
            rows, rejects = [], []
            for topic, payload in batch:
                msg, reason = validate(topic, payload)
                if msg is None:
                    rejects.append((topic, reason, payload[:120].decode("utf-8", "replace")))
                    self.reject_reasons[reason.split(":")[0]] = self.reject_reasons.get(reason.split(":")[0], 0) + 1
                else:
                    msg["error_codes"] = json.dumps(msg["error_codes"])
                    rows.append(tuple(msg[c] for c in COLS))
            before = con.total_changes
            con.executemany(INSERT, rows)
            con.executemany("INSERT INTO rejected VALUES (?,?,?)", rejects)
            con.commit()
            stored = con.total_changes - before - len(rejects)
            self.stats["received"] += len(batch)
            self.stats["stored"] += stored
            self.stats["duplicates"] += len(rows) - stored
            self.stats["rejected"] += len(rejects)
            self.last_ts = time.time()
        con.close()

    def wait_until(self, n_received: int, timeout: float = 120.0, idle_s: float = 4.0) -> bool:
        """Wait for n_received messages; give up early if nothing new arrives for idle_s seconds."""
        deadline = time.time() + timeout
        last, last_change = self.stats["received"], time.time()
        while time.time() < deadline:
            now_n = self.stats["received"]
            if now_n >= n_received:
                return True
            if now_n != last:
                last, last_change = now_n, time.time()
            elif time.time() - last_change > idle_s:
                return False
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
        self._stop.set()
        self._writer.join(timeout=30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/fleet.db")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    svc = Ingest(args.db, args.port).start()
    print(f"ingest listening on 127.0.0.1:{args.port}, writing {args.db} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(5)
            print(svc.stats, flush=True)
    except KeyboardInterrupt:
        svc.stop()
