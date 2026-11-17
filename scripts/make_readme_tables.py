"""Fill the results section of README.md from results/*.json. No number is typed by hand.

Run:  uv run python scripts/make_readme_tables.py
"""
import json
import re
import sys

R = "results"
load = lambda n: json.load(open(f"{R}/{n}"))  # noqa: E731
f = lambda x, d=2: "n/a" if x is None else f"{x:.{d}f}"  # noqa: E731


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ing, ctl = load("ingest_report.json"), load("ingest_report_default_timeout.json")
    m, cv, par, fw = load("metrics.json"), load("cv_summary.json"), load("pareto.json"), load("firmware_grouping.json")
    chk, rca = load("leakage_checks.json"), load("rca_test_devices.json")
    P = m["protocol"]
    s = []
    s.append(f"### Ingest (`results/ingest_report.json`)\n")
    s.append(md(["Quantity", "Value"], [
        ["Devices x simulated days", f"{ing['n_devices']} x {ing['horizon_days']}"],
        ["Messages published (QoS 1), PUBACKs received", f"{ing['messages_published']}, {ing['puback_received']}"],
        ["Deliberately corrupted payloads injected / rejected by validator",
         f"{ing['bad_payloads_injected']} / {ing['ingest_rejected']} ({ing['reject_reasons']})"],
        ["Valid messages / stored after first pass / lost in transit (first pass)",
         f"{ing['valid_payloads']} / {ing['first_pass']['stored']} / {ing['first_pass']['valid_lost_in_transit']} "
         f"(loss rate {ing['first_pass']['loss_rate']})"],
        ["Gap-fill rounds (republished, stored after)",
         "; ".join(f"{r['republished']}, {r['stored_after']}" for r in ing["gapfill_rounds"]) or "none needed"],
        ["Rows missing after gap-fill (of all generated)", ing["final_missing_after_gapfill"]],
        ["First-pass end-to-end throughput", f"{ing['first_pass_throughput_msgs_per_s']} msgs/s ({ing['first_pass_end_to_end_seconds']} s)"],
    ]))
    s.append(f"\nNegative control with amqtt's stock 5 s QoS 1 ack timeout and no gap-fill "
             f"(`results/ingest_report_default_timeout.json`): {ctl['valid_payloads']} valid sent, "
             f"{ctl['first_pass']['stored']} stored, {ctl['first_pass']['valid_lost_in_transit']} lost "
             f"(loss rate {ctl['first_pass']['loss_rate']}), {ctl['first_pass_throughput_msgs_per_s']} msgs/s.\n")

    s.append(f"### Failure-mode metrics on {P['test_devices']} held-out devices (`results/metrics.json`)\n")
    s.append(f"Train: {P['train_devices']} devices, test: {P['test_devices']} devices "
             f"(test composition: {P['test_modes']}). No device appears in both.\n")
    rows = []
    for mode in ["healthy", "brush_hair_wrap", "clogged_filter", "battery_degradation", "wheel_wear",
                 "firmware_regression"]:
        r = []
        for key in ("rule_baseline", "gradient_boosting"):
            pm = m[key]["per_mode"][mode]
            r += [f(pm["precision"]), f(pm["recall"]), f(pm["f1"])]
        rows.append([mode, m["gradient_boosting"]["per_mode"][mode]["support"]] + r)
    s.append(md(["Mode", "n test devices", "Rule P", "Rule R", "Rule F1", "GB P", "GB R", "GB F1"], rows))
    s.append(f"\nMacro-F1 over the 5 fault modes: rule baseline {f(m['rule_baseline']['macro_f1_fault_modes'], 3)}, "
             f"gradient boosting {f(m['gradient_boosting']['macro_f1_fault_modes'], 3)}. "
             f"GB names the right mode on the first alert day for "
             f"{f(100 * m['gradient_boosting']['mode_correct_on_first_alert_day']['fraction'], 1)} % of alerted faulty devices "
             f"({m['gradient_boosting']['mode_correct_on_first_alert_day']['devices']}).\n")

    s.append("### Detection and lead time on the same test devices\n")
    rows = []
    for label, key in [("Rule baseline", "rule_baseline"), ("Isolation Forest (unsupervised)", "isolation_forest_unsupervised"),
                       ("Gradient boosting", "gradient_boosting")]:
        d, lt = m[key]["detection"], m[key]["lead_time"]["all_degrading"]
        rows.append([label, f(d["any_fault_recall"]), f(d["any_fault_precision"]),
                     f"{d['healthy_devices_alerted']}/{d['healthy_devices']}",
                     lt["alerted_before_hard_failure"], f(lt["lead_days_median"], 1), f(lt["lead_days_p90"], 1)])
    s.append(md(["Detector", "Fault recall (any mode)", "Precision", "Healthy devices alerted",
                 "Alerted before hard failure (of 76 degrading)", "Lead median (days)", "Lead p90 (days)"], rows))
    s.append("\nLead time = days between the first alert and the injected hard-failure day, over degrading-mode "
             "devices alerted before it. Firmware regressions have no hard-failure day.\n")
    s.append("Per-mode lead time for gradient boosting:\n")
    lt = m["gradient_boosting"]["lead_time"]
    s.append(md(["Mode", "Devices", "Alerted before failure", "Median (d)", "p90 (d)"],
                [[k, lt[k]["devices"], lt[k]["alerted_before_hard_failure"], f(lt[k]["lead_days_median"], 1),
                  f(lt[k]["lead_days_p90"], 1)] for k in ("brush_hair_wrap", "clogged_filter",
                                                          "battery_degradation", "wheel_wear")]))

    cm = m["gradient_boosting"]["confusion_matrix"]
    s.append("\nGradient-boosting confusion matrix (rows = true, columns = predicted):\n")
    s.append(md(["true \\ pred"] + cm["labels_true_rows_pred_cols"],
                [[l] + r for l, r in zip(cm["labels_true_rows_pred_cols"], cm["matrix"])]))

    s.append("\n### 5-fold device-level cross-validation (`results/cv_summary.json`)\n")
    rows = []
    for name, label in [("rule", "Rule baseline"), ("iforest", "Isolation Forest"), ("gb", "Gradient boosting")]:
        sm = cv["summary"][name]
        g = lambda k: f"{sm[k]['mean']:.3f} +/- {sm[k]['std']:.3f}" if k in sm else "n/a"  # noqa: E731
        rows.append([label, g("macro_f1_fault_modes"), g("any_fault_recall"), g("healthy_false_alarm_rate"), g("lead_median")])
    s.append(md(["Detector", "Macro-F1 (fault modes)", "Fault recall", "Healthy false-alarm rate", "Median lead (d)"], rows))

    s.append("\n### Root cause (`results/pareto.json`, `results/rca_test_devices.json`, `results/firmware_grouping.json`)\n")
    s.append("Fleet Pareto, out-of-fold predictions over all devices:\n")
    s.append(md(["Predicted mode", "Devices", "Share", "Cumulative"],
                [[p["mode"], p["devices"], f(p["share"]), f(p["cumulative_share"])] for p in par["predicted"]]))
    s.append(f"\nInjected truth for comparison: {par['injected_truth_devices']}.\n")
    s.append("Reboot errors by firmware version (grouping, telemetry only):\n")
    s.append(md(["Firmware", "Devices", "Runs", "Reboots / 100 runs", "95% CI"],
                [[r["firmware"], r["devices"], r["runs"], r["reboots_per_100_runs"],
                  f"{r['ci95_low_per_100']}-{r['ci95_high_per_100']}"] for r in fw["by_firmware"]]))
    s.append(f"\n{fw['highest_rate_firmware']} shows {fw['rate_ratio_vs_all_other_versions']}x the reboot rate of the other versions.\n")
    hr = rca["hit_rate"]
    s.append(f"Explanation check: for correctly-moded flagged test devices, a physically expected signal is among "
             f"the top 3 in {hr['overall']['hit']}/{hr['overall']['of']} cases "
             f"({f(100 * hr['overall']['rate'], 1)} %). Top signal groups by permutation importance: "
             f"{', '.join(list(rca['permutation_importance']['balanced_accuracy_drop_by_signal'])[:4])}.\n")

    s.append("### Leakage and easiness checks (`results/leakage_checks.json`)\n")
    rows = [["Device overlap train/test", chk["split_overlap_devices"], "", ""],
            ["Ground-truth columns used as features", len(chk["ground_truth_columns_in_model_features"]), "", ""]]
    for k in ("shuffled_training_labels", "no_firmware_features", "no_error_or_firmware_features", "no_slope_features"):
        c = chk[k]
        rows.append([k, f(c["macro_f1_fault_modes"], 3), f(c["any_fault_recall"]), f(c["healthy_false_alarm_rate"])])
    s.append(md(["Check", "GB macro-F1 (fault modes) / count", "Fault recall", "Healthy false-alarm rate"], rows))
    s.append("\nBy injected fault strength (GB, test devices):\n")
    s.append(md(["Strength", "Devices", "Alerted", "Correct mode"],
                [[k, v["devices"], f(v["alerted_share"]), f(v["correct_mode_share"])]
                 for k, v in chk["gb_by_fault_strength"].items()]))
    body = "\n".join(s)
    text = open("README.md").read()
    new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                 lambda _: f"<!-- RESULTS:START -->\n{body}\n<!-- RESULTS:END -->", text, flags=re.S)
    open("README.md", "w").write(new)
    print("README results section regenerated", file=sys.stderr)


if __name__ == "__main__":
    main()
