"""Technical-indicator and feature-engineering utilities.

This module provides the :class:`FeatureEngineer` class which augments a raw
OHLCV :class:`~pandas.DataFrame` with a standard set of technical indicators
used downstream by trading strategies and ML pipelines.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("High", "Low", "Close")


class FeatureEngineer:
    """Compute technical indicators and engineered features from OHLCV data.

    The engineer is configurable via indicator look-back windows and produces
    the following columns:

    * ``log_return`` -- daily log returns of the close price.
    * ``sma_20`` -- 20-day simple moving average of the close.
    * ``ema_12`` / ``ema_26`` -- 12- and 26-day exponential moving averages.
    * ``rsi_14`` -- 14-day Relative Strength Index (Wilder's smoothing).
    * ``atr_14`` -- 14-day Average True Range (Wilder's smoothing).

    Missing data is handled explicitly: input rows are forward-filled, and any
    rows still containing NaNs after indicator computation are dropped. The
    number of rows altered by each step is logged.

    Attributes:
        sma_window: Window length for the simple moving average.
        ema_short: Span for the short exponential moving average.
        ema_long: Span for the long exponential moving average.
        rsi_window: Look-back window for RSI.
        atr_window: Look-back window for ATR.
    """

    def __init__(
        self,
        sma_window: int = 20,
        ema_short: int = 12,
        ema_long: int = 26,
        rsi_window: int = 14,
        atr_window: int = 14,
    ) -> None:
        """Initialize the feature engineer.

        Args:
            sma_window: Window for the simple moving average (must be >= 1).
            ema_short: Span for the short EMA (must be >= 1).
            ema_long: Span for the long EMA (must be >= 1).
            rsi_window: Window for RSI (must be >= 1).
            atr_window: Window for ATR (must be >= 1).

        Raises:
            ValueError: If any window/span parameter is < 1.
        """
        params = {
            "sma_window": sma_window,
            "ema_short": ema_short,
            "ema_long": ema_long,
            "rsi_window": rsi_window,
            "atr_window": atr_window,
        }
        for name, value in params.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        self.sma_window: int = sma_window
        self.ema_short: int = ema_short
        self.ema_long: int = ema_long
        self.rsi_window: int = rsi_window
        self.atr_window: int = atr_window

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Augment an OHLCV DataFrame with technical indicators.

        The input is not mutated; a new DataFrame is returned. Missing values
        in the input are forward-filled before indicators are computed, and any
        rows still containing NaNs (e.g. the indicator warm-up period) are
        dropped from the output.

        Args:
            data: A DataFrame containing at least the columns ``High``,
                ``Low`` and ``Close``, indexed chronologically.

        Returns:
            A new DataFrame containing the original columns plus the engineered
            feature columns, with warm-up/NaN rows removed.

        Raises:
            ValueError: If ``data`` is empty or missing a required column.
        """
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        missing = [c for c in _REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing columns: {missing}")

        df = data.copy()

        n_missing_before = int(df[list(_REQUIRED_COLUMNS)].isna().sum().sum())
        if n_missing_before:
            df = df.ffill()
            logger.info(
                "Forward-filled %d missing values in required columns",
                n_missing_before,
            )

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        df["log_return"] = np.log(close / close.shift(1))
        df["sma_20"] = close.rolling(window=self.sma_window).mean()
        df["ema_12"] = close.ewm(span=self.ema_short, adjust=False).mean()
        df["ema_26"] = close.ewm(span=self.ema_long, adjust=False).mean()
        df["rsi_14"] = self._compute_rsi(close, self.rsi_window)
        df["atr_14"] = self._compute_atr(high, low, close, self.atr_window)

        rows_before = len(df)
        df = df.dropna()
        rows_dropped = rows_before - len(df)
        logger.info(
            "Dropped %d rows containing NaNs after feature computation "
            "(%d -> %d rows)",
            rows_dropped,
            rows_before,
            len(df),
        )
        return df

    @staticmethod
    def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
        """Compute the Relative Strength Index using Wilder's smoothing.

        Args:
            close: Series of closing prices.
            window: Look-back window length.

        Returns:
            A Series of RSI values in the range ``[0, 100]``.
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        # Wilder's smoothing is an EMA with alpha = 1 / window.
        avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # When average loss is zero, RSI is defined as 100.
        rsi = rsi.where(avg_loss != 0.0, 100.0)
        # When both are zero (flat series), RSI is neutral 50.
        rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
        return rsi

    @staticmethod
    def _compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Compute the Average True Range using Wilder's smoothing.

        Args:
            high: Series of high prices.
            low: Series of low prices.
            close: Series of closing prices.
            window: Look-back window length.

        Returns:
            A Series of ATR values.
        """
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1.0 / window, min_periods=window).mean()
