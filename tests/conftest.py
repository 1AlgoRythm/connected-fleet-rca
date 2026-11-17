import json
import os

import pytest

from fleet.run_transport import run


@pytest.fixture(scope="session")
def small_fleet(tmp_path_factory):
    """Whole transport path (broker + publisher + ingest) on a small fleet; shared by tests."""
    d = tmp_path_factory.mktemp("fleet")
    report = run(str(d / "fleet.db"), str(d), port=18841, n_devices=60, horizon=60, n_bad=8)
    truth = json.load(open(os.path.join(d, "ground_truth.json")))
    return {"db": str(d / "fleet.db"), "report": report, "truth": truth, "dir": str(d)}
