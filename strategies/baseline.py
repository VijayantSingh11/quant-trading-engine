"""Baseline trend-following strategy implementation."""

from __future__ import annotations

from typing import Final

import pandas as pd

from strategies.base import BaseStrategy

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("ema_12", "ema_26", "rsi_14")


class TrendFollowingStrategy(BaseStrategy):
    """An EMA-crossover trend-following strategy with an RSI filter.

    The strategy goes long when the short EMA is above the long EMA (an
    established up-trend) and short when the short EMA is below the long EMA.
    An RSI filter suppresses entries into over-extended conditions: longs are
    only taken while RSI is below ``rsi_overbought`` and shorts only while RSI
    is above ``rsi_oversold``. When neither condition holds the strategy stays
    flat (in cash).

    Attributes:
        rsi_overbought: Upper RSI threshold above which new longs are blocked.
        rsi_oversold: Lower RSI threshold below which new shorts are blocked.
    """

    def __init__(
        self,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> None:
        """Initialize the trend-following strategy.

        Args:
            rsi_overbought: RSI level (0-100) above which longs are filtered
                out. Must be greater than ``rsi_oversold``.
            rsi_oversold: RSI level (0-100) below which shorts are filtered
                out.

        Raises:
            ValueError: If thresholds are out of range or inconsistent.
        """
        if not 0.0 <= rsi_oversold < rsi_overbought <= 100.0:
            raise ValueError(
                "Require 0 <= rsi_oversold < rsi_overbought <= 100, got "
                f"oversold={rsi_oversold}, overbought={rsi_overbought}"
            )
        self.rsi_overbought: float = rsi_overbought
        self.rsi_oversold: float = rsi_oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate position signals from EMA crossovers filtered by RSI.

        Args:
            data: A DataFrame containing the columns ``ema_12``, ``ema_26``
                and ``rsi_14``, indexed chronologically.

        Returns:
            An integer :class:`~pandas.Series` aligned to ``data``'s index with
            values in ``{-1, 0, 1}`` (short, flat, long).

        Raises:
            ValueError: If any required feature column is missing.
        """
        self._validate_columns(data, _REQUIRED_COLUMNS)

        ema_short = data["ema_12"]
        ema_long = data["ema_26"]
        rsi = data["rsi_14"]

        long_trend = ema_short > ema_long
        short_trend = ema_short < ema_long

        long_ok = long_trend & (rsi < self.rsi_overbought)
        short_ok = short_trend & (rsi > self.rsi_oversold)

        signals = pd.Series(0, index=data.index, dtype="int64")
        signals[long_ok] = 1
        signals[short_ok] = -1
        return signals
