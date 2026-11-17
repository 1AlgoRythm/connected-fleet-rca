"""Broker + publisher + ingest + SQLite, end to end, on 60 devices x 60 days."""
import sqlite3


def test_every_valid_message_is_stored_and_bad_ones_rejected(small_fleet):
    r = small_fleet["report"]
    assert r["puback_received"] == r["messages_published"]            # broker acked every QoS 1 publish
    assert r["bad_payloads_injected"] == 8
    assert r["ingest_rejected"] == 8
    # first pass may lose messages on the broker->subscriber leg (observed with amqtt); gap-fill repairs it
    assert r["first_pass"]["valid_lost_in_transit"] >= 0
    assert r["final_missing_after_gapfill"] == 0
    assert r["distinct_rows_stored"] == r["messages_generated"]


def test_rejected_rows_have_reasons(small_fleet):
    con = sqlite3.connect(small_fleet["db"])
    reasons = {x[0] for x in con.execute("SELECT reason FROM rejected")}
    con.close()
    assert reasons <= {"bad_value:suction_a", "missing:battery_v_min", "bad_value:runtime_min",
                       "topic_device_mismatch"}
    assert len(reasons) >= 3
