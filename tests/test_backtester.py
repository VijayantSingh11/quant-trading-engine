"""Unit tests for the historical backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.engine import HistoricalBacktester


def _price_frame(prices: list[float]) -> pd.DataFrame:
    """Build a minimal Close-only frame from a price list."""
    index = pd.date_range("2022-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=index)


def test_flat_signals_preserve_capital() -> None:
    """With no positions taken, equity must stay at the starting capital."""
    data = _price_frame([100.0, 101.0, 102.0, 103.0])
    signals = pd.Series(0, index=data.index)
    bt = HistoricalBacktester(initial_capital=100_000.0)
    result = bt.run(data, signals)
    assert np.allclose(result.equity_curve.to_numpy(), 100_000.0)
    assert result.trades == []


def test_long_profits_on_rising_market_no_fees() -> None:
    """A long position over a rising market should grow equity (no fees)."""
    data = _price_frame([100.0, 110.0, 120.0, 130.0])
    # Signal on day 0 -> position from day 1 onward (look-ahead-free).
    signals = pd.Series([1, 1, 1, 1], index=data.index)
    bt = HistoricalBacktester(
        initial_capital=100_000.0, fee_pct=0.0, position_size=1.0
    )
    result = bt.run(data, signals)
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]
    assert len(result.trades) == 1
    assert result.trades[0].pnl > 0


def test_no_lookahead_first_bar_is_flat() -> None:
    """The first bar is always flat because signals are lagged by one."""
    data = _price_frame([100.0, 110.0, 120.0])
    signals = pd.Series([1, 1, 1], index=data.index)
    bt = HistoricalBacktester(initial_capital=100_000.0, fee_pct=0.0)
    result = bt.run(data, signals)
    assert result.positions.iloc[0] == 0
    assert result.equity_curve.iloc[0] == pytest.approx(100_000.0)


def test_fees_reduce_equity_on_round_trip() -> None:
    """Entering and exiting flat market should lose exactly the fees."""
    data = _price_frame([100.0, 100.0, 100.0, 100.0])
    # Go long from bar 1, then flat from bar 3 (signal flips at bar 2).
    signals = pd.Series([1, 1, 0, 0], index=data.index)
    bt = HistoricalBacktester(
        initial_capital=100_000.0, fee_pct=0.01, position_size=1.0
    )
    result = bt.run(data, signals)
    # Two sides of 1% fee on ~100k notional => ~2% lost.
    assert result.equity_curve.iloc[-1] < 100_000.0
    assert result.equity_curve.iloc[-1] == pytest.approx(98_010.0, rel=1e-3)


def test_short_profits_on_falling_market() -> None:
    """A short position over a falling market should grow equity."""
    data = _price_frame([100.0, 90.0, 80.0, 70.0])
    signals = pd.Series([-1, -1, -1, -1], index=data.index)
    bt = HistoricalBacktester(
        initial_capital=100_000.0, fee_pct=0.0, position_size=1.0
    )
    result = bt.run(data, signals)
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]


def test_index_mismatch_raises() -> None:
    """Mismatched data/signal indices should raise ValueError."""
    data = _price_frame([100.0, 101.0])
    signals = pd.Series([0, 0], index=pd.date_range("2000-01-01", periods=2))
    with pytest.raises(ValueError):
        HistoricalBacktester().run(data, signals)


def test_missing_close_raises() -> None:
    """A frame without a Close column should raise ValueError."""
    index = pd.date_range("2022-01-01", periods=2, freq="B")
    data = pd.DataFrame({"Open": [1.0, 2.0]}, index=index)
    signals = pd.Series([0, 0], index=index)
    with pytest.raises(ValueError):
        HistoricalBacktester().run(data, signals)


def test_invalid_constructor_params() -> None:
    """Out-of-range constructor arguments should raise ValueError."""
    with pytest.raises(ValueError):
        HistoricalBacktester(initial_capital=0)
    with pytest.raises(ValueError):
        HistoricalBacktester(fee_pct=1.0)
    with pytest.raises(ValueError):
        HistoricalBacktester(position_size=0)
