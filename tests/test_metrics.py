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
