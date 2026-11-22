"""Early-detection metrics.

All functions take numpy arrays in packet sequence for ONE capture.
The delay unit is packets, because the feature files have no time column.
"""

from __future__ import annotations

import numpy as np


def find_episodes(labels: np.ndarray, merge_gap: int = 0) -> np.ndarray:
    """Find attack episodes in a 0/1 label array.

    An episode is a run of malicious packets. Runs that have fewer than
    ``merge_gap`` benign packets between them become one episode, because
    benign packets also flow during an attack.

    Returns an int64 array of shape (n_episodes, 2): [start, end), end exclusive.
    """
    labels = np.asarray(labels).astype(np.int8)
    if labels.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    padded = np.concatenate(([0], labels, [0]))
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    if starts.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if merge_gap > 0 and starts.size > 1:
        gaps = starts[1:] - ends[:-1]
        keep = gaps >= merge_gap
        starts = np.concatenate((starts[:1], starts[1:][keep]))
        ends = np.concatenate((ends[:-1][keep], ends[-1:]))
    return np.stack((starts, ends), axis=1).astype(np.int64)


def alerts_k_of_n(scores: np.ndarray, threshold: float, k: int = 1, n: int = 1) -> np.ndarray:
    """Alert at packet i if at least k of the last n scores are more than the threshold.

    The window includes packet i and uses only past packets, so the rule can
    run online. k=1, n=1 gives a plain threshold.
    """
    if not 1 <= k <= n:
        raise ValueError("k and n must obey 1 <= k <= n")
    hits = (np.asarray(scores) > threshold).astype(np.int64)
    csum = np.concatenate(([0], np.cumsum(hits)))
    idx = np.arange(1, hits.size + 1)
    window = csum[idx] - csum[np.maximum(idx - n, 0)]
    return window >= k


def detection_delays(alerts: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    """Delay in packets from each episode start to the first alert in the episode.

    Returns a float array with one value for each episode. NaN = missed episode.
    """
    alert_idx = np.flatnonzero(np.asarray(alerts))
    delays = np.full(len(episodes), np.nan)
    for i, (start, end) in enumerate(episodes):
        pos = np.searchsorted(alert_idx, start, side="left")
        if pos < alert_idx.size and alert_idx[pos] < end:
            delays[i] = alert_idx[pos] - start
    return delays


def false_alarm_stats(alerts: np.ndarray, labels: np.ndarray) -> dict:
    """False alarms: alerts on packets that have a benign label."""
    alerts = np.asarray(alerts, dtype=bool)
    benign = np.asarray(labels) == 0
    n_benign = int(benign.sum())
    n_false = int((alerts & benign).sum())
    rate = n_false / n_benign if n_benign else float("nan")
    return {
        "benign_packets": n_benign,
        "false_alarms": n_false,
        "false_positive_rate": rate,
        "false_alarms_per_100k": rate * 100_000,
    }


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    k: int = 1,
    n: int = 1,
    merge_gap: int = 0,
) -> dict:
    """Full early-detection summary for one capture and one alert setting."""
    labels = np.asarray(labels)
    alerts = alerts_k_of_n(scores, threshold, k, n)
    episodes = find_episodes(labels, merge_gap)
    delays = detection_delays(alerts, episodes)
    detected = ~np.isnan(delays)
    malicious = labels == 1
    n_malicious = int(malicious.sum())
    out = {
        "threshold": threshold,
        "k": k,
        "n": n,
        "episodes": int(len(episodes)),
        "episodes_detected": int(detected.sum()),
        "mean_delay_packets": float(delays[detected].mean()) if detected.any() else float("nan"),
        "median_delay_packets": float(np.median(delays[detected])) if detected.any() else float("nan"),
        "max_delay_packets": float(delays[detected].max()) if detected.any() else float("nan"),
        "packet_recall": float((alerts & malicious).sum() / n_malicious) if n_malicious else float("nan"),
    }
    out.update(false_alarm_stats(alerts, labels))
    return out


def credibility(
    delay_packets,
    false_alarms_before_attack: int,
    benign_packets_before_attack: int,
    packet_recall: float,
    min_lift: float = 5.0,
    max_chance: float = 0.05,
) -> dict:
    """Tell a real detection from an alert that false alarms can explain.

    A detector that gives false alarms at a constant rate also gives an
    "alert" soon after each attack start. Two checks prevent that error:

    alert_lift
        Alert rate on malicious packets, divided by the alert rate on benign
        packets before the attack. ``detection_credible`` is true if the lift
        is at least ``min_lift``: the detector reacts to the attack.

    chance_delay_probability
        Probability that a detector with no skill, and the same false alarm
        rate, gives an alert in the same number of packets or fewer.
        ``delay_credible`` is true if the detection is credible and this
        probability is at most ``max_chance``: the delay shows early detection.

    The benign alert rate has a floor of one alert in the benign period.
    Zero false alarms in a short period is thus not proof of a zero rate.
    """
    if not benign_packets_before_attack:
        return {"alert_lift": float("nan"), "detection_credible": False,
                "chance_delay_probability": float("nan"), "delay_credible": False}
    benign_rate = max(false_alarms_before_attack, 1) / benign_packets_before_attack
    lift = packet_recall / benign_rate if packet_recall == packet_recall else float("nan")
    detection_ok = bool(lift >= min_lift)
    detected = delay_packets is not None and delay_packets == delay_packets
    if detected:
        chance = float(-np.expm1((delay_packets + 1) * np.log1p(-min(benign_rate, 1 - 1e-12))))
    else:
        chance = float("nan")
    return {
        "alert_lift": float(lift),
        "detection_credible": detection_ok,
        "chance_delay_probability": chance,
        "delay_credible": bool(detected and detection_ok and chance <= max_chance),
    }


def add_credibility(results, min_lift: float = 5.0, max_chance: float = 0.05):
    """Add the credibility columns to a results table (pandas DataFrame)."""
    import pandas as pd

    rows = [
        credibility(r.delay_packets, r.false_alarms_before_attack, r.benign_packets_before_attack,
                    r.packet_recall, min_lift, max_chance)
        for r in results.itertuples()
    ]
    extra = pd.DataFrame(rows, index=results.index)
    return pd.concat([results.drop(columns=[c for c in extra.columns if c in results.columns]), extra], axis=1)


def onset_evaluation(
    scores: np.ndarray,
    labels: np.ndarray,
    onset: int,
    threshold: float,
    k: int = 1,
    n: int = 1,
    min_lift: float = 5.0,
    max_chance: float = 0.05,
) -> dict:
    """Evaluate one capture that has one attack start at position ``onset``.

    False alarms are counted only before the attack start. An alert on a
    benign packet during the attack is not a false alarm for the operator.
    """
    labels = np.asarray(labels)
    alerts = alerts_k_of_n(scores, threshold, k, n)
    after = np.flatnonzero(alerts[onset:])
    detected = after.size > 0
    false_alarms = int(alerts[:onset].sum())
    malicious = labels == 1
    n_malicious = int(malicious.sum())
    delay = int(after[0]) if detected else None
    recall = float((alerts & malicious).sum() / n_malicious) if n_malicious else float("nan")
    return {
        "threshold": threshold,
        "k": k,
        "n": n,
        "detected": bool(detected),
        "delay_packets": delay,
        "benign_packets_before_attack": int(onset),
        "false_alarms_before_attack": false_alarms,
        "false_alarms_per_100k": false_alarms / onset * 100_000 if onset else float("nan"),
        "packet_recall": recall,
        **credibility(delay, false_alarms, onset, recall, min_lift, max_chance),
    }


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """ROC-AUC and PR-AUC. These do not use a threshold."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }
