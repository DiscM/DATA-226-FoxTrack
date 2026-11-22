import numpy as np
import pytest

from foxtrack.metrics import alerts_k_of_n, find_episodes, moving_mean
from foxtrack.serving import OnlineDetector, PcaModel


def make_model(n_features=6, k=2, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n_features, k)))
    thresholds = [
        {"target_fpr": 1e-3, "alert_rule": "threshold", "smooth_window": 1, "k": 1, "n": 1, "threshold": 1.0},
        {"target_fpr": 1e-3, "alert_rule": "3 of 5", "smooth_window": 1, "k": 3, "n": 5, "threshold": 1.0},
        {"target_fpr": 1e-3, "alert_rule": "mean of 100", "smooth_window": 100, "k": 1, "n": 1, "threshold": 0.5},
    ]
    return PcaModel("demo", rng.normal(size=n_features), np.ones(n_features), q.T, thresholds)


def traffic(model, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    x = model.mean + rng.normal(size=(n, model.n_features)) * 0.5
    x[2000:] += 3.0
    return x


def test_score_is_reconstruction_error():
    model = make_model()
    x = traffic(model, 50)
    z = x - model.mean
    expected = ((z - z @ model.components.T @ model.components) ** 2).sum(1) / model.n_features
    assert np.allclose(model.score(x), expected)
    with pytest.raises(ValueError):
        model.score(np.zeros((2, 3)))


def test_zero_variance_feature_gives_zero():
    model = make_model()
    model.std[0] = 0.0
    x = traffic(model, 10)
    x[:, 0] = 1e9
    assert np.isfinite(model.score(x)).all()


def test_unknown_setting():
    with pytest.raises(KeyError):
        make_model().rule("mean of 100", 0.5)


@pytest.mark.parametrize("rule", ["threshold", "3 of 5", "mean of 100"])
@pytest.mark.parametrize("chunk", [1, 7, 100, 3000])
def test_online_detector_is_equal_to_one_batch(rule, chunk):
    model = make_model()
    x = traffic(model)
    scores = model.score(x)
    setting = model.rule(rule, 1e-3)
    if setting["smooth_window"] > 1:
        expected = moving_mean(scores, setting["smooth_window"]) > setting["threshold"]
    else:
        expected = alerts_k_of_n(scores, setting["threshold"], setting["k"], setting["n"])

    detector = OnlineDetector(model, rule, 1e-3)
    alerts = np.concatenate([detector.update(x[i:i + chunk])["alerts"] for i in range(0, len(x), chunk)])
    assert alerts.tolist() == expected.tolist()
    assert detector.packets_seen == len(x)
    assert detector.alert_packets == int(expected.sum())
    assert detector.alert_events == len(find_episodes(expected, merge_gap=100))
    assert expected[2100:].mean() > 0.9  # the attack is detected
