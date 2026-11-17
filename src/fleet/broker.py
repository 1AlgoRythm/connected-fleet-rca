"""Pure-Python MQTT broker (amqtt) so the pipeline needs no system install.

Run standalone:  uv run python -m fleet.broker --port 18830
Or embed:        with BrokerProcess(port) as b: ...   (used by tests and pipeline)
"""
from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import socket
import time

from amqtt.broker import Broker

DEFAULT_PORT = 18830  # unprivileged and unlikely to clash with a real mosquitto on 1883


def broker_config(port: int, puback_timeout: float | None = None) -> dict:
    return {
        "listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{port}"}},
        # Only anonymous auth: it is a localhost test broker. No packet loggers
        # (they would print every message and dominate the throughput we measure).
        "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}},
        "qos1_puback_timeout": puback_timeout,   # None = wait for subscriber acks (amqtt default: drop after 5 s)
    }


async def _serve(port: int, puback_timeout: float | None) -> None:
    broker = Broker(broker_config(port, puback_timeout))
    await broker.start()
    try:
        await asyncio.Event().wait()  # run until the process is terminated
    finally:
        await broker.shutdown()


def serve_forever(port: int, puback_timeout: float | None = None) -> None:
    asyncio.run(_serve(port, puback_timeout))


def wait_for_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"broker did not open port {port} within {timeout}s")


class BrokerProcess:
    """Context manager: broker in a child process (keeps its event loop off ours)."""

    def __init__(self, port: int = DEFAULT_PORT, puback_timeout: float | None = None):
        self.port, self.puback_timeout = port, puback_timeout
        self._proc: mp.Process | None = None

    def __enter__(self) -> "BrokerProcess":
        ctx = mp.get_context("spawn")
        self._proc = ctx.Process(target=serve_forever, args=(self.port, self.puback_timeout), daemon=True)
        self._proc.start()
        wait_for_port(self.port)
        return self

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(timeout=5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_forever(ap.parse_args().port)
