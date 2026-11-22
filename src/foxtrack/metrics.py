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
    false_alarm_events: int | None = None,
) -> dict:
    """Tell a real detection from an alert that false alarms can explain.

    A detector that gives false alarms also gives an "alert" soon after each
    attack start. Two checks prevent that error:

    alert_lift
        Fraction of malicious packets with an alert, divided by the fraction
        of benign packets with an alert before the attack.
        ``detection_credible`` is true if the lift is at least ``min_lift``:
        the detector reacts to the attack.

    chance_delay_probability
        Probability that a detector with no skill, and the same false alarms,
        gives an alert in the same number of packets or fewer.
        ``delay_credible`` is true if the detection is credible and this
        probability is at most ``max_chance``: the delay shows early detection.

    The chance model. False alarms come as events. An event is a sequence of
    alerts, and it can continue for many packets (a moving mean stays high
    for the length of its window). An attack start at a random time gets an
    alert with no delay if an event is in progress (probability = fraction of
    benign packets with an alert). If not, the next event starts at the rate
    of events for each benign packet. If ``false_alarm_events`` is not given,
    each false alarm packet is one event.

    The two rates have a floor of one in the benign period. Zero false alarms
    in a short period is thus not proof of a zero rate.
    """
    if not benign_packets_before_attack:
        return {"alert_lift": float("nan"), "detection_credible": False,
                "chance_delay_probability": float("nan"), "delay_credible": False}
    n = benign_packets_before_attack
    events = false_alarms_before_attack if false_alarm_events is None else false_alarm_events
    alert_fraction = min(max(false_alarms_before_attack, 1) / n, 1.0)
    event_rate = max(events, 1) / n
    lift = packet_recall / alert_fraction if packet_recall == packet_recall else float("nan")
    detection_ok = bool(lift >= min_lift)
    detected = delay_packets is not None and delay_packets == delay_packets
    if detected:
        chance = float(1.0 - (1.0 - alert_fraction) * np.exp(-event_rate * delay_packets))
    else:
        chance = float("nan")
    return {
        "alert_lift": float(lift),
        "detection_credible": detection_ok,
        "chance_delay_probability": chance,
        "delay_credible": bool(detected and detection_ok and chance <= max_chance),
    }


def rate_upper_limit(events: int, packets: int, confidence: float = 0.95) -> float:
    """Upper confidence limit of an event rate (events for each packet), Poisson model.

    Zero events in N packets does not prove a zero rate. With 95% confidence
    the rate is less than approximately 3 / N.
    """
    from scipy.stats import chi2

    if not packets:
        return float("nan")
    return float(chi2.ppf(confidence, 2 * (events + 1)) / 2 / packets)


def add_credibility(results, min_lift: float = 5.0, max_chance: float = 0.05):
    """Add the credibility columns to a results table (pandas DataFrame).

    Uses the column ``false_alarm_events`` if the table has it.
    """
    import pandas as pd

    has_events = "false_alarm_events" in results.columns
    rows = [
        credibility(r.delay_packets, r.false_alarms_before_attack, r.benign_packets_before_attack,
                    r.packet_recall, min_lift, max_chance, r.false_alarm_events if has_events else None)
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
    event_gap: int = 100,
) -> dict:
    """Evaluate one capture that has one attack start at position ``onset``.

    A false alarm event is a sequence of alerts before the attack. Alerts
    with fewer than ``event_gap`` packets between them are one event. An
    operator gets one alarm for one event, so the event count is the
    operational measure. The packet count is much larger for a rule that
    stays in alert for many packets, such as a moving mean.

    False alarms are counted only before the attack start. An alert on a
    benign packet during the attack is not a false alarm for the operator.
    """
    labels = np.asarray(labels)
    alerts = alerts_k_of_n(scores, threshold, k, n)
    after = np.flatnonzero(alerts[onset:])
    detected = after.size > 0
    false_alarms = int(alerts[:onset].sum())
    events = int(len(find_episodes(alerts[:onset], merge_gap=event_gap)))
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
        "false_alarm_events": events,
        "false_alarm_events_per_100k": events / onset * 100_000 if onset else float("nan"),
        "false_alarm_events_per_100k_upper95": rate_upper_limit(events, onset) * 100_000,
        "packet_recall": recall,
        **credibility(delay, false_alarms, onset, recall, min_lift, max_chance, events),
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


def moving_mean(scores: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` values of log(1 + score), for each packet.

    One benign packet with a high score gives a false alarm. The mean removes
    such single peaks, but an attack keeps the mean high. The window uses only
    past packets, so the rule can run online. The logarithm limits the effect
    of one extreme score. ``window`` = 1 returns the scores with no change.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if window < 1:
        raise ValueError("window must be 1 or more")
    if window == 1:
        return scores
    csum = np.concatenate(([0.0], np.cumsum(np.log1p(np.maximum(scores, 0.0)))))
    idx = np.arange(1, scores.size + 1)
    start = np.maximum(idx - window, 0)
    return (csum[idx] - csum[start]) / (idx - start)


def alert_rule_name(k: int, n: int, smooth_window: int) -> str:
    if smooth_window > 1:
        return f"mean of {smooth_window}"
    return "threshold" if (k, n) == (1, 1) else f"{k} of {n}"


PLACEBO_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def placebo_credible_count(benign_scores: np.ndarray, threshold: float, k: int = 1, n: int = 1,
                           fractions=PLACEBO_FRACTIONS) -> int:
    """Number of false "attack starts" in benign traffic that the check accepts as credible.

    Each placebo start is a position in the benign test packets. The packets
    after it get the label malicious, although no attack is there. A correct
    procedure must find no credible early detection. A count of more than
    zero shows that the result for this setting is not reliable.
    """
    benign_scores = np.asarray(benign_scores)
    count = 0
    for fraction in fractions:
        start = int(benign_scores.size * fraction)
        if start == 0 or start >= benign_scores.size:
            continue
        labels = np.zeros(benign_scores.size, dtype=np.int8)
        labels[start:] = 1
        count += onset_evaluation(benign_scores, labels, start, threshold, k, n)["delay_credible"]
    return int(count)


def evaluate_grid(
    calib_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    onset: int,
    target_fprs=(1e-2, 1e-3, 1e-4),
    alert_rules=((1, 1), (3, 5), (10, 20)),
    smooth_windows=(1, 100),
) -> list[dict]:
    """Evaluate one capture for each threshold and each alert rule.

    ``calib_scores`` are benign packets immediately before ``test_scores``,
    the two in packet sequence. The threshold for a target false positive
    rate comes from the calibration packets.

    Alert rules: a plain threshold, k of the last n scores (``alert_rules``),
    and a moving mean of the score (each window more than 1 in
    ``smooth_windows``). The threshold for a moving mean comes from the moving
    mean of the calibration packets, so its unit is different.
    """
    calib_scores = np.asarray(calib_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    rows = []
    for window in smooth_windows:
        smoothed = moving_mean(np.concatenate((calib_scores, test_scores)), window)
        calib, test = smoothed[: calib_scores.size], smoothed[calib_scores.size:]
        rules = alert_rules if window == 1 else ((1, 1),)
        for fpr in target_fprs:
            threshold = float(np.quantile(calib, 1 - fpr))
            for k, n in rules:
                result = onset_evaluation(test, labels, onset, threshold, k, n)
                placebo = placebo_credible_count(test[:onset], threshold, k, n)
                rows.append({"target_fpr": fpr, "alert_rule": alert_rule_name(k, n, window),
                             "smooth_window": window, **result,
                             "placebo_credible": placebo, "placebo_starts": len(PLACEBO_FRACTIONS)})
    return rows
