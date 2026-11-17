# STUDY: how this project works, end to end

Read this before an interview. Numbers are not repeated here on purpose: every number lives in `results/*.json` and the README tables generated from them. If a number in this file ever disagrees with those, the JSON is right.

## What this is, in one paragraph

A fake fleet of 600 robot vacuums "runs" for up to 90 days. Each run produces a small JSON report (motor currents, battery voltage, airflow, error codes, firmware version). The reports go over MQTT to a broker, a subscriber validates them and stores them in SQLite, and then machine learning looks at each device's own history to say "this one is going wrong, it looks like a brush hair wrap, and the signal that gave it away is the brush motor current". Because I injected the faults myself, I know the right answers and can score the methods. **It is host-simulated: no board, no sensors, no real data.**

## Which file to read first

1. `src/fleet/simulate.py` (the physics, and the failures I inject). Read `make_fleet` then `simulate_device`.
2. `src/fleet/schema.py` and `src/fleet/ingest.py` (what the cloud side trusts and why).
3. `src/fleet/run_transport.py` (how the pieces are wired and how loss is measured and repaired).
4. `src/fleet/features.py` (the key idea: compare each device with itself).
5. `src/fleet/detectors.py`, `src/fleet/evaluate.py` (three detectors, one fair way to judge them).
6. `src/fleet/explain.py` (root cause, Pareto, firmware grouping).
7. `tests/` (what is guaranteed).

## The data flow in plain words

* **Simulator.** Each device has fixed personality traits (how strong its motors are, how much carpet it sees, whether pets live there) and a daily story (hair builds up and is cleared, the filter clogs and is washed, the battery slowly ages). A fault adds an extra effect that grows from nothing at the onset day to full at the hard-failure day. Units are in the field names: A, V, mAh, L/s, min, m^2.
* **Publish.** Each message goes to topic `fleet/<device_id>/telemetry` at QoS 1. QoS 1 means "at least once": the broker acknowledges receipt, and a message may even arrive twice. That is why SQLite has `PRIMARY KEY (device_id, run_id)` and `INSERT OR IGNORE`.
* **Validate.** `schema.py` checks types, physical ranges, and that the topic's device id equals the payload's device id (a device may only speak for itself). Bad messages go to a `rejected` table with a reason; they are never silently dropped.
* **Features.** One row per device per day. Each signal is divided by that device's own first-10-day average, then averaged over the last 7 days, plus 14-day slopes and error rates. Only the past is used.
* **Detectors.** (a) A rule baseline with fixed limits; (b) an Isolation Forest that never sees labels; (c) gradient boosting that predicts which mode (or healthy). All three are judged the same way: a device alerts after 3 consecutive flagged days, and each threshold is set on training devices so at most about 5 % of healthy ones ever alert.
* **Root cause.** For a flagged device, hide one signal at a time (replace it by the healthy median) and see how much the predicted-mode probability falls. The biggest drops are the "top signals". A Pareto counts flagged devices by predicted mode. Grouping reboot errors by firmware version shows the firmware regression without any ML.

## What I measured and how to read it

See the README "Results" section. Things to say out loud in an interview:

* The split is by device, never by row. A row-level split would put neighbouring days of the same device in train and test and flatter the model.
* The broker acknowledging the publisher does not prove the subscriber got the message. In my runs messages sometimes disappeared on the broker-to-subscriber leg (I did not find the root cause inside the amqtt broker). I measured that as "first-pass loss", then added a **gap-fill**: devices number their runs, the cloud finds the missing sequence numbers, devices resend. `results/ingest_report.json` shows the first-pass numbers and what gap-fill repaired.
* An earlier race in my own code (publishing before the subscription was acknowledged) also dropped messages; `Ingest.start()` now waits for the SUBACK.
* I first generated a simulation in which the classifier was nearly perfect, which usually means the task is too easy. I made the healthy world messier (heavier-tailed dirt, hair and battery ageing, floor-mix changes, weaker faults) once and did not tune further. The leakage checks in `results/leakage_checks.json` (shuffled labels, ablations, performance by fault strength) are there so I can say what the result does and does not show. The first, easier version was not kept as a result file, so I cite no numbers for it.

## Interview questions and short answers

1. **Why MQTT and not HTTP?** Small battery-powered devices with flaky Wi-Fi: MQTT keeps one long-lived connection, has tiny headers, supports QoS levels and per-topic fan-out. HTTP is request/response and heavier per message.
2. **What does QoS 1 guarantee and what does it not?** At-least-once delivery per hop: the sender retries until acknowledged, so duplicates are possible and order across devices is not guaranteed. It does not guarantee the other hop worked; I saw this in practice.
3. **How do you handle duplicates and loss?** Idempotent writes on `(device_id, run_id)`; a per-device sequence number lets the cloud detect gaps and request a resend (gap-fill).
4. **Why validate at ingest?** Devices run old firmware, bytes get corrupted, and a bad value in a table poisons analytics later. Validation makes failure explicit (rejected table with reasons) and enforces that a device only publishes under its own id.
5. **Why divide each signal by the device's own baseline?** Healthy units differ (motor spread, carpet vs hard floor). A fleet-wide absolute threshold either misses faults or alarms on healthy outliers. The cost: it needs a clean baseline period, which a real fleet must guarantee.
6. **Why split by device?** The model must work on units it has never seen. Rows from one device are near-duplicates of each other, so a row split leaks.
7. **How do you know the model is not cheating?** Features contain no ground-truth columns (asserted); hard-failure error codes are excluded; features are causal (tested by truncation); shuffled training labels collapse the score; ablations show which feature families matter; metrics are broken down by fault strength.
8. **How is lead time computed, and what are its limits?** Hard-failure day minus first-alert day, only for devices alerted before the failure. It depends on my simulator's fault ramp (linear, 20-40 days) and on the 3-day debounce; real faults are not that clean.
9. **Why Isolation Forest if gradient boosting is better?** Unsupervised needs no labels, which you do not have for new failure types. It can say "unusual" but not "which mode". It is the right first line when field returns are not yet labelled.
10. **What does firmware grouping show that ML does not?** A version-level cause. Reboot rate by firmware version isolates the bad build immediately and tells you what to roll back; a per-device classifier only tells you which units are affected.
11. **How reliable is the "top signals" explanation?** It is an occlusion test on a trained model: it shows what the model used, which for correlated signals can differ from the physical cause. I checked it against the simulator's known physics (how often the expected signal is in the top 3), not against real teardown data.
12. **What would you do differently on real hardware?** Use real field-return teardown labels, handle clock skew and offline buffering on the device, add OTA staged rollout metrics (so the regression is caught at 1 % of the fleet), calibrate the false-alarm budget against repair cost, and add a firmware-side PID/health check for the motors.

## Known limits

* All data is simulated by code I wrote, with effect sizes I chose. Good scores prove the method works on that world, not on real robots.
* The per-device verdict uses data after the first alert (hindsight); the first-alert-day mode accuracy is the stricter number.
* Only one fault per device, one severity ramp shape per mode, all faults start after the baseline window.
* 600 devices, 90 days, a single seed for the main split plus 5-fold CV; confidence intervals on per-mode numbers are wide because each mode has only about 19 test devices.
* amqtt is a convenience broker, not a production broker. Throughput numbers are one process on one laptop.
* The Streamlit screenshot (if present in `docs/`) shows simulated data; the dashboard shows out-of-fold verdicts.
* The GitHub Actions workflow `.github/workflows/ci.yml` has never run (no remote exists).
