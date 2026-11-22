import numpy as np
import pytest

from foxtrack.metrics import (
    alerts_k_of_n,
    detection_delays,
    evaluate,
    false_alarm_stats,
    find_episodes,
)


def test_find_episodes_basic():
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    assert find_episodes(labels).tolist() == [[2, 4], [5, 6]]


def test_find_episodes_edges_and_empty():
    assert find_episodes(np.array([1, 1, 0, 1])).tolist() == [[0, 2], [3, 4]]
    assert find_episodes(np.zeros(5)).shape == (0, 2)
    assert find_episodes(np.array([])).shape == (0, 2)


def test_find_episodes_merge_gap():
    labels = np.array([1, 0, 1, 0, 0, 0, 1])
    assert find_episodes(labels, merge_gap=2).tolist() == [[0, 3], [6, 7]]
    assert find_episodes(labels, merge_gap=4).tolist() == [[0, 7]]


def test_alerts_plain_threshold():
    scores = np.array([0.1, 0.9, 0.5, 0.6])
    assert alerts_k_of_n(scores, 0.5).tolist() == [False, True, False, True]


def test_alerts_k_of_n_uses_only_past_packets():
    scores = np.array([0, 1, 0, 1, 1, 0, 0, 0])
    # 2 of the last 3 scores must be high.
    assert alerts_k_of_n(scores, 0.5, k=2, n=3).tolist() == [
        False, False, False, True, True, True, False, False,
    ]


def test_alerts_bad_arguments():
    with pytest.raises(ValueError):
        alerts_k_of_n(np.zeros(3), 0.5, k=3, n=2)


def test_detection_delays():
    alerts = np.zeros(20, dtype=bool)
    alerts[[1, 7, 15]] = True
    episodes = np.array([[5, 10], [12, 14], [15, 20]])
    delays = detection_delays(alerts, episodes)
    assert delays[0] == 2
    assert np.isnan(delays[1])  # the alert at 15 is after this episode
    assert delays[2] == 0


def test_false_alarm_stats():
    alerts = np.array([1, 0, 1, 1], dtype=bool)
    labels = np.array([0, 0, 1, 0])
    stats = false_alarm_stats(alerts, labels)
    assert stats["false_alarms"] == 2
    assert stats["false_positive_rate"] == pytest.approx(2 / 3)


def test_evaluate_summary():
    labels = np.array([0] * 10 + [1] * 10)
    scores = np.array([0.0] * 13 + [1.0] * 7)
    out = evaluate(scores, labels, threshold=0.5)
    assert out["episodes"] == 1
    assert out["episodes_detected"] == 1
    assert out["mean_delay_packets"] == 3
    assert out["false_alarms"] == 0
    assert out["packet_recall"] == pytest.approx(0.7)


def test_credibility_real_detection():
    from foxtrack.metrics import credibility
    # 10 false alarms in 100,000 benign packets, 90% recall, alert after 3 packets.
    out = credibility(3, 10, 100_000, 0.9)
    assert out["alert_lift"] == pytest.approx(9000)
    assert out["detection_credible"] and out["delay_credible"]
    assert out["chance_delay_probability"] == pytest.approx(1 - (1 - 1e-4) ** 4)


def test_credibility_rejects_alert_at_false_alarm_rate():
    from foxtrack.metrics import credibility
    # Recall is equal to the false alarm rate: the detector does not react to the attack.
    out = credibility(46, 155, 100_000, 0.002)
    assert out["alert_lift"] == pytest.approx(1.29, abs=0.01)
    assert not out["detection_credible"] and not out["delay_credible"]


def test_credibility_rejects_late_alert():
    from foxtrack.metrics import credibility
    # The detector reacts to the attack, but false alarms alone explain the delay.
    out = credibility(4310, 120, 100_000, 0.5)
    assert out["detection_credible"]
    assert out["chance_delay_probability"] > 0.99
    assert not out["delay_credible"]


def test_credibility_edge_cases():
    from foxtrack.metrics import credibility
    missed = credibility(None, 0, 1000, 0.0)
    assert not missed["detection_credible"] and not missed["delay_credible"]
    assert np.isnan(missed["chance_delay_probability"])
    # Zero false alarms: the rate has a floor of one alert in the benign period.
    clean = credibility(0, 0, 1000, 1.0)
    assert clean["alert_lift"] == pytest.approx(1000)
    assert clean["chance_delay_probability"] == pytest.approx(0.001)
    assert clean["delay_credible"]
    assert not credibility(5, 0, 0, 1.0)["delay_credible"]


def test_onset_evaluation_has_credibility_columns():
    from foxtrack.metrics import onset_evaluation
    labels = np.array([0] * 1000 + [1] * 100)
    scores = labels.astype(float)
    out = onset_evaluation(scores, labels, onset=1000, threshold=0.5)
    assert out["delay_credible"] and out["delay_packets"] == 0


def test_add_credibility_to_results_table():
    import pandas as pd
    from foxtrack.metrics import add_credibility
    table = pd.DataFrame({
        "delay_packets": [3.0, np.nan],
        "false_alarms_before_attack": [10, 0],
        "benign_packets_before_attack": [100_000, 100_000],
        "packet_recall": [0.9, 0.0],
    })
    out = add_credibility(add_credibility(table))  # a second call must not add duplicate columns
    assert out["delay_credible"].tolist() == [True, False]
    assert list(out.columns).count("alert_lift") == 1


def test_moving_mean_uses_only_past_packets():
    from foxtrack.metrics import moving_mean
    scores = np.array([0.0, 0.0, np.e - 1, 0.0, 0.0])  # log1p gives 0, 0, 1, 0, 0
    assert moving_mean(scores, 2).tolist() == pytest.approx([0, 0, 0.5, 0.5, 0])
    assert moving_mean(scores, 3)[:3].tolist() == pytest.approx([0, 0, 1 / 3])
    assert moving_mean(scores, 1) is not None and moving_mean(scores, 1).tolist() == scores.tolist()
    with pytest.raises(ValueError):
        moving_mean(scores, 0)


def test_evaluate_grid_moving_mean_removes_single_peaks():
    from foxtrack.metrics import evaluate_grid
    rng = np.random.default_rng(0)
    # Benign: low scores with rare single peaks. Attack: a moderate, sustained increase.
    calib = rng.random(5000) * 0.1
    calib[::500] = 5.0
    benign = rng.random(5000) * 0.1
    benign[::500] = 5.0
    attack = 1.0 + rng.random(2000) * 0.1
    test = np.concatenate((benign, attack))
    labels = np.concatenate((np.zeros(5000, int), np.ones(2000, int)))

    rows = evaluate_grid(calib, test, labels, onset=5000, target_fprs=(1e-3,), smooth_windows=(1, 50))
    by_rule = {r["alert_rule"]: r for r in rows}
    assert set(by_rule) == {"threshold", "3 of 5", "10 of 20", "mean of 50"}
    # The plain threshold is set by the peaks, so it does not see the moderate attack.
    assert by_rule["threshold"]["packet_recall"] == 0
    smooth = by_rule["mean of 50"]
    assert smooth["false_alarms_before_attack"] == 0
    assert smooth["packet_recall"] > 0.95
    assert smooth["delay_packets"] < 50
    assert smooth["smooth_window"] == 50 and smooth["delay_credible"]


def test_evaluate_grid_window_one_is_the_plain_threshold():
    from foxtrack.metrics import evaluate_grid, onset_evaluation
    rng = np.random.default_rng(1)
    calib, test = rng.random(1000), np.concatenate((rng.random(500), rng.random(500) + 1))
    labels = np.concatenate((np.zeros(500, int), np.ones(500, int)))
    row = evaluate_grid(calib, test, labels, 500, target_fprs=(1e-2,), alert_rules=((1, 1),), smooth_windows=(1,))[0]
    direct = onset_evaluation(test, labels, 500, float(np.quantile(calib, 0.99)))
    assert {k: row[k] for k in direct} == direct
