from bandobast.forecast import EscalationForecaster
from bandobast.synth import generate_scenario


def test_forecast_beats_persistence():
    f = EscalationForecaster(seed=42).fit()
    report = f.report(generate_scenario(seed=42))
    assert report.beats_persistence
    assert report.rmse < report.persistence_rmse


def test_forecast_deterministic():
    r1 = EscalationForecaster(seed=11).fit().report(generate_scenario(seed=11))
    r2 = EscalationForecaster(seed=11).fit().report(generate_scenario(seed=11))
    assert r1.rmse == r2.rmse
    assert [x.predicted_delta for x in r1.forecasts] == [x.predicted_delta for x in r2.forecasts]


def test_forecast_reports_importances_and_calibration():
    report = EscalationForecaster(seed=5).fit().report(generate_scenario(seed=5))
    assert len(report.feature_importances) == 6
    assert len(report.calibration) > 0
