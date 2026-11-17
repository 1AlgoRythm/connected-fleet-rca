import copy
import json

from fleet.schema import validate
from fleet.simulate import simulate_fleet

MSG = simulate_fleet(n_devices=20, horizon=10)[0][0]
TOPIC = f"fleet/{MSG['device_id']}/telemetry"


def _check(mutate):
    m = copy.deepcopy(MSG)
    mutate(m)
    return validate(TOPIC, json.dumps(m).encode())


def test_good_message_passes():
    msg, reason = validate(TOPIC, json.dumps(MSG).encode())
    assert reason is None and msg["device_id"] == MSG["device_id"]


def test_rejections():
    assert _check(lambda m: m.pop("brush_a"))[1] == "missing:brush_a"
    assert _check(lambda m: m.update(runtime_min=-1))[1] == "bad_value:runtime_min"
    assert _check(lambda m: m.update(suction_a="NaN"))[1] == "bad_value:suction_a"
    assert _check(lambda m: m.update(device_id="SV-9999"))[1] == "topic_device_mismatch"
    assert _check(lambda m: m.update(error_codes="E_X"))[1] == "bad_type:error_codes"
    assert validate(TOPIC, b"{not json")[1] == "not_json"
    assert validate("fleet/x/other", json.dumps(MSG).encode())[1] == "topic_device_mismatch"
