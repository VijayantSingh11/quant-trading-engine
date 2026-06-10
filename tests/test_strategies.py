"""Unit tests for the trading strategies."""

from __future__ import annotations

import pandas as pd
import pytest

from features.engineer import FeatureEngineer
from strategies.base import BaseStrategy
from strategies.baseline import TrendFollowingStrategy


def test_base_strategy_is_abstract() -> None:
    """BaseStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseStrategy()  # type: ignore[abstract]


def test_signals_only_contain_valid_values(ohlcv_frame: pd.DataFrame) -> None:
    """Generated signals must only ever be -1, 0, or 1."""
    features = FeatureEngineer().transform(ohlcv_frame)
    signals = TrendFollowingStrategy().generate_signals(features)
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert signals.index.equals(features.index)


def test_long_signal_when_ema_cross_up_and_rsi_ok() -> None:
    """Short EMA above long EMA with neutral RSI should yield a long."""
    index = pd.date_range("2022-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "ema_12": [10.0, 11.0, 12.0],
            "ema_26": [9.0, 9.5, 10.0],
            "rsi_14": [50.0, 55.0, 60.0],
        },
        index=index,
    )
    signals = TrendFollowingStrategy().generate_signals(df)
    assert (signals == 1).all()


def test_rsi_filter_blocks_overbought_long() -> None:
    """An up-trend with overbought RSI should not produce a long signal."""
    index = pd.date_range("2022-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        {"ema_12": [12.0, 13.0], "ema_26": [10.0, 10.5], "rsi_14": [80.0, 85.0]},
        index=index,
    )
    signals = TrendFollowingStrategy().generate_signals(df)
    assert (signals == 0).all()


def test_short_signal_when_ema_cross_down() -> None:
    """Short EMA below long EMA with mid RSI should yield a short."""
    index = pd.date_range("2022-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        {"ema_12": [8.0, 7.0], "ema_26": [10.0, 10.0], "rsi_14": [50.0, 45.0]},
        index=index,
    )
    signals = TrendFollowingStrategy().generate_signals(df)
    assert (signals == -1).all()


def test_invalid_thresholds_raise() -> None:
    """Inconsistent RSI thresholds should raise ValueError."""
    with pytest.raises(ValueError):
        TrendFollowingStrategy(rsi_overbought=30.0, rsi_oversold=70.0)


def test_missing_feature_column_raises() -> None:
    """Missing a required feature column should raise ValueError."""
    df = pd.DataFrame({"ema_12": [1.0], "ema_26": [2.0]})
    with pytest.raises(ValueError):
        TrendFollowingStrategy().generate_signals(df)
