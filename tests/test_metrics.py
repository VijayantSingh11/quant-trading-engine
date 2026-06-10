"""Unit tests for the RiskAnalytics performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from performance.metrics import RiskAnalytics


def _equity(values: list[float]) -> pd.Series:
    """Build an equity-curve Series from a list of values."""
    index = pd.date_range("2022-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=index, name="equity")


def test_total_return() -> None:
    """Total return should reflect first-to-last growth."""
    metrics = RiskAnalytics().analyze(_equity([100.0, 150.0]))
    assert metrics["total_return"] == pytest.approx(0.5)


def test_max_drawdown() -> None:
    """Max drawdown should capture the largest peak-to-trough decline."""
    curve = _equity([100.0, 120.0, 60.0, 90.0])
    metrics = RiskAnalytics().analyze(curve)
    # Peak 120 -> trough 60 = -50%.
    assert metrics["max_drawdown"] == pytest.approx(-0.5)


def test_no_drawdown_for_monotonic_curve() -> None:
    """A strictly increasing curve has zero drawdown."""
    metrics = RiskAnalytics().analyze(_equity([100.0, 110.0, 120.0, 130.0]))
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_volatility_zero_for_constant_returns() -> None:
    """A constant growth rate yields (near) zero volatility."""
    curve = _equity([100.0 * (1.01 ** i) for i in range(50)])
    metrics = RiskAnalytics().analyze(curve)
    assert metrics["annualized_volatility"] == pytest.approx(0.0, abs=1e-9)


def test_sharpe_uses_risk_free_rate() -> None:
    """Sharpe ratio should be finite and float-typed for a noisy curve."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 252)
    curve = _equity(list(100.0 * np.cumprod(1 + rets)))
    metrics = RiskAnalytics(risk_free_rate=0.04).analyze(curve)
    assert np.isfinite(metrics["sharpe_ratio"])


def test_win_loss_ratio_from_trade_pnls() -> None:
    """Win/loss ratio should derive from trade PnLs when provided."""
    curve = _equity([100.0, 101.0, 102.0])
    metrics = RiskAnalytics().analyze(
        curve, trade_pnls=[10.0, -5.0, 3.0, -2.0]
    )
    assert metrics["win_loss_ratio"] == pytest.approx(1.0)


def test_win_loss_ratio_all_wins_is_inf() -> None:
    """With only winners, the win/loss ratio is infinite."""
    curve = _equity([100.0, 110.0, 120.0])
    metrics = RiskAnalytics().analyze(curve, trade_pnls=[5.0, 7.0])
    assert metrics["win_loss_ratio"] == float("inf")


def test_short_curve_raises() -> None:
    """An equity curve with fewer than two points should raise ValueError."""
    with pytest.raises(ValueError):
        RiskAnalytics().analyze(_equity([100.0]))


def test_non_positive_equity_raises() -> None:
    """A non-positive equity value should raise ValueError."""
    with pytest.raises(ValueError):
        RiskAnalytics().analyze(_equity([100.0, 0.0, 50.0]))
