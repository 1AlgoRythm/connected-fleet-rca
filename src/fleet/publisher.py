"""Device-side publisher: pushes simulated run telemetry to the broker at QoS 1.

One MQTT client per shard of devices (real devices each hold their own connection;
a few clients is the host-side compromise). Time is compressed: simulated weeks go
out as fast as the broker acks, the payload `ts` keeps the simulated clock.
"""
from __future__ import annotations

import json
import threading
import time
import zlib

import paho.mqtt.client as mqtt

from .broker import DEFAULT_PORT


def corrupt(msg: dict, k: int) -> dict:
    """Deterministic bad payloads, to prove the validator rejects them."""
    bad = dict(msg)
    kind = k % 4
    if kind == 0:
        bad["suction_a"] = "NaN"            # wrong type
    elif kind == 1:
        bad.pop("battery_v_min")            # missing field
    elif kind == 2:
        bad["runtime_min"] = -5             # physically impossible
    else:
        bad["device_id"] = "SV-9999"        # topic/device mismatch
    return bad


def publish_all(messages: list[dict], port: int = DEFAULT_PORT, n_clients: int = 4,
                n_bad: int = 0) -> dict:
    """Publish every message; return counts and timing. Replaces `n_bad` messages by corrupted ones."""
    shards: list[list[tuple[str, bytes]]] = [[] for _ in range(n_clients)]
    bad_every = max(1, len(messages) // n_bad) if n_bad else 0
    n_bad_sent = 0
    for i, msg in enumerate(messages):
        if bad_every and i % bad_every == 0 and n_bad_sent < n_bad:
            msg = corrupt(msg, n_bad_sent)
            n_bad_sent += 1
        topic = f"fleet/{messages[i]['device_id']}/telemetry"
        # all of one device's messages go through one client: keeps per-device ordering
        shards[zlib.crc32(messages[i]["device_id"].encode()) % n_clients].append((topic, json.dumps(msg).encode()))
    sent, acked = [0] * n_clients, [0] * n_clients

    def work(idx: int) -> None:
        cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"fleet-pub-{idx}")
        cli.max_inflight_messages_set(500)
        cli.on_publish = lambda c, u, mid, rc, p=None: acked.__setitem__(idx, acked[idx] + 1)
        cli.connect("127.0.0.1", port)
        cli.loop_start()
        infos = [cli.publish(t, p, qos=1) for t, p in shards[idx]]
        sent[idx] = len(infos)
        for info in infos:
            info.wait_for_publish(60)
        cli.loop_stop()
        cli.disconnect()

    t0 = time.time()
    threads = [threading.Thread(target=work, args=(i,)) for i in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"messages_published": sum(sent), "puback_received": sum(acked),
            "bad_payloads_injected": n_bad_sent, "valid_payloads": sum(sent) - n_bad_sent,
            "publish_seconds": round(time.time() - t0, 3), "publisher_clients": n_clients}
