"""Unit tests for the FeatureEngineer indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.engineer import FeatureEngineer


def test_transform_adds_expected_columns(ohlcv_frame: pd.DataFrame) -> None:
    """All engineered feature columns should be present after transform."""
    engineer = FeatureEngineer()
    out = engineer.transform(ohlcv_frame)
    for col in ("log_return", "sma_20", "ema_12", "ema_26", "rsi_14", "atr_14"):
        assert col in out.columns
    assert not out.isna().any().any()


def test_transform_does_not_mutate_input(ohlcv_frame: pd.DataFrame) -> None:
    """The input DataFrame must not be modified in place."""
    original = ohlcv_frame.copy()
    FeatureEngineer().transform(ohlcv_frame)
    pd.testing.assert_frame_equal(ohlcv_frame, original)


def test_sma_matches_manual_rolling_mean(ohlcv_frame: pd.DataFrame) -> None:
    """SMA column should equal a manual 20-day rolling mean of Close."""
    out = FeatureEngineer(sma_window=20).transform(ohlcv_frame)
    expected = ohlcv_frame["Close"].rolling(20).mean().loc[out.index]
    pd.testing.assert_series_equal(
        out["sma_20"], expected, check_names=False
    )


def test_rsi_bounded_between_0_and_100(ohlcv_frame: pd.DataFrame) -> None:
    """RSI values must always lie within [0, 100]."""
    out = FeatureEngineer().transform(ohlcv_frame)
    assert out["rsi_14"].between(0.0, 100.0).all()


def test_rsi_is_100_for_monotonic_increasing_series() -> None:
    """A strictly increasing close series should drive RSI to 100."""
    n = 60
    index = pd.date_range("2021-01-01", periods=n, freq="B")
    close = pd.Series(np.arange(1, n + 1, dtype=float), index=index)
    df = pd.DataFrame(
        {"High": close + 0.5, "Low": close - 0.5, "Close": close}
    )
    out = FeatureEngineer().transform(df)
    assert np.allclose(out["rsi_14"].to_numpy(), 100.0)


def test_atr_is_positive(ohlcv_frame: pd.DataFrame) -> None:
    """ATR must be strictly positive for a noisy price series."""
    out = FeatureEngineer().transform(ohlcv_frame)
    assert (out["atr_14"] > 0).all()


def test_forward_fill_counts_missing(ohlcv_frame: pd.DataFrame) -> None:
    """Internal NaNs should be forward-filled rather than dropped early."""
    frame = ohlcv_frame.copy()
    frame.iloc[100, frame.columns.get_loc("Close")] = np.nan
    out = FeatureEngineer().transform(frame)
    assert not out["Close"].isna().any()


def test_empty_frame_raises() -> None:
    """An empty input frame should raise ValueError."""
    with pytest.raises(ValueError):
        FeatureEngineer().transform(pd.DataFrame())


def test_missing_column_raises(ohlcv_frame: pd.DataFrame) -> None:
    """A frame missing a required column should raise ValueError."""
    with pytest.raises(ValueError):
        FeatureEngineer().transform(ohlcv_frame.drop(columns=["Close"]))


def test_invalid_window_raises() -> None:
    """Constructing with a non-positive window should raise ValueError."""
    with pytest.raises(ValueError):
        FeatureEngineer(rsi_window=0)
