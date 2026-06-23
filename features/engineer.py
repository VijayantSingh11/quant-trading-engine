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

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("High", "Low", "Close", "Volume")


class FeatureEngineer:
    """Compute technical indicators and engineered features from OHLCV data.

    The engineer is configurable via indicator look-back windows and produces
    the following columns:

    * ``log_return`` -- daily log returns of the close price.
    * ``sma_20`` -- 20-day simple moving average of the close.
    * ``ema_12`` / ``ema_26`` -- 12- and 26-day exponential moving averages.
    * ``rsi_14`` -- 14-day Relative Strength Index (Wilder's smoothing).
    * ``macd`` -- MACD line (ema_12 minus ema_26).
    * ``macd_signal`` -- 9-day EMA of the MACD line.
    * ``bb_width`` -- Bollinger Band Width: (upper - lower) / middle.
    * ``atr_14`` -- 14-day Average True Range (Wilder's smoothing).
    * ``atr_pct`` -- ATR normalised by close price (ATR / Close).
    * ``vol_ratio`` -- Current volume divided by 20-day volume SMA.
    * ``target_dir`` -- Binary forward-return target (1 if 5-day log return >
      0, else 0). Lagged correctly so there is **zero look-ahead bias**.

    Missing data is handled explicitly: input rows are forward-filled, and any
    rows still containing NaNs after indicator computation are dropped. The
    number of rows altered by each step is logged.

    Attributes:
        sma_window: Window length for the simple moving average.
        ema_short: Span for the short exponential moving average.
        ema_long: Span for the long exponential moving average.
        rsi_window: Look-back window for RSI.
        atr_window: Look-back window for ATR.
        macd_signal_span: EMA span for the MACD signal line.
        bb_window: Window for Bollinger Bands.
        bb_num_std: Number of standard deviations for Bollinger Bands.
        vol_sma_window: Window for the volume SMA used in the vol ratio.
        forward_return_window: Number of bars ahead for the target variable.
    """

    def __init__(
        self,
        sma_window: int = 20,
        ema_short: int = 12,
        ema_long: int = 26,
        rsi_window: int = 14,
        atr_window: int = 14,
        macd_signal_span: int = 9,
        bb_window: int = 20,
        bb_num_std: float = 2.0,
        vol_sma_window: int = 20,
        forward_return_window: int = 5,
    ) -> None:
        """Initialize the feature engineer.

        Args:
            sma_window: Window for the simple moving average (must be >= 1).
            ema_short: Span for the short EMA (must be >= 1).
            ema_long: Span for the long EMA (must be >= 1).
            rsi_window: Window for RSI (must be >= 1).
            atr_window: Window for ATR (must be >= 1).
            macd_signal_span: EMA span for the MACD signal line (must be >= 1).
            bb_window: Rolling window for Bollinger Bands (must be >= 2).
            bb_num_std: Number of standard deviations for band width.
            vol_sma_window: Window for the volume SMA ratio (must be >= 1).
            forward_return_window: Number of bars ahead used to build the
                binary target variable (must be >= 1).

        Raises:
            ValueError: If any window/span parameter is < 1, or bb_window < 2.
        """
        int_params = {
            "sma_window": sma_window,
            "ema_short": ema_short,
            "ema_long": ema_long,
            "rsi_window": rsi_window,
            "atr_window": atr_window,
            "macd_signal_span": macd_signal_span,
            "vol_sma_window": vol_sma_window,
            "forward_return_window": forward_return_window,
        }
        for name, value in int_params.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if bb_window < 2:
            raise ValueError(f"bb_window must be >= 2, got {bb_window}")

        self.sma_window: int = sma_window
        self.ema_short: int = ema_short
        self.ema_long: int = ema_long
        self.rsi_window: int = rsi_window
        self.atr_window: int = atr_window
        self.macd_signal_span: int = macd_signal_span
        self.bb_window: int = bb_window
        self.bb_num_std: float = float(bb_num_std)
        self.vol_sma_window: int = vol_sma_window
        self.forward_return_window: int = forward_return_window

    def transform(self, data: pd.DataFrame, add_target: bool = False) -> pd.DataFrame:
        """Augment an OHLCV DataFrame with technical indicators.

        The input is not mutated; a new DataFrame is returned. Missing values
        in the input are forward-filled before indicators are computed, and any
        rows still containing NaNs (e.g. the indicator warm-up period) are
        dropped from the output.

        Args:
            data: A DataFrame containing at least the columns ``High``,
                ``Low``, ``Close`` and ``Volume``, indexed chronologically.
            add_target: If ``True``, compute the binary ``target_dir`` column
                using the forward-return window. Rows where the target cannot
                be computed (the final ``forward_return_window`` bars) are
                dropped. Defaults to ``False`` so that inference-time
                transforms do not require future data.

        Returns:
            A new DataFrame containing the original columns plus the engineered
            feature columns, with warm-up/NaN rows removed.

        Raises:
            ValueError: If ``data`` is empty or missing a required column.
        """
        required = _REQUIRED_COLUMNS if "Volume" in data.columns else _REQUIRED_COLUMNS[:3]
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        missing = [c for c in ("High", "Low", "Close") if c not in data.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing columns: {missing}")

        df = data.copy()

        n_missing_before = int(df[["High", "Low", "Close"]].isna().sum().sum())
        if n_missing_before:
            df = df.ffill()
            logger.info(
                "Forward-filled %d missing values in required columns",
                n_missing_before,
            )

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ── Core indicators (existing) ────────────────────────────────────────
        df["log_return"] = np.log(close / close.shift(1))
        df["sma_20"] = close.rolling(window=self.sma_window).mean()
        df["ema_12"] = close.ewm(span=self.ema_short, adjust=False).mean()
        df["ema_26"] = close.ewm(span=self.ema_long, adjust=False).mean()
        df["rsi_14"] = self._compute_rsi(close, self.rsi_window)
        df["atr_14"] = self._compute_atr(high, low, close, self.atr_window)

        # ── Momentum: MACD & MACD Signal ────────────────────────────────────
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(
            span=self.macd_signal_span, adjust=False
        ).mean()

        # ── Volatility: Bollinger Band Width & ATR% ──────────────────────────
        roll = close.rolling(window=self.bb_window)
        bb_middle = roll.mean()
        bb_std = roll.std(ddof=1)
        bb_upper = bb_middle + self.bb_num_std * bb_std
        bb_lower = bb_middle - self.bb_num_std * bb_std
        # Bandwidth: (upper - lower) / middle; avoids division issues on flat
        df["bb_width"] = (bb_upper - bb_lower) / bb_middle.replace(0.0, np.nan)
        df["atr_pct"] = df["atr_14"] / close.replace(0.0, np.nan)

        # ── Volume: Current Volume / 20-day Volume SMA ───────────────────────
        if "Volume" in df.columns:
            vol_sma = df["Volume"].rolling(window=self.vol_sma_window).mean()
            df["vol_ratio"] = df["Volume"] / vol_sma.replace(0.0, np.nan)
        else:
            df["vol_ratio"] = np.nan
            logger.warning("Volume column absent; vol_ratio set to NaN")

        # ── Target variable (only during training) ───────────────────────────
        if add_target:
            # Forward log return over next `forward_return_window` bars.
            fwd_log_ret = np.log(
                close.shift(-self.forward_return_window) / close
            )
            # Convert to binary target: 1 if positive, 0 otherwise.
            # Use pd.NA-aware mapping so that tail NaNs remain NaN and are
            # subsequently dropped by the dropna() call below -- this is the
            # mechanism that prevents look-ahead bias.
            target = fwd_log_ret.map(
                lambda x: 1 if (x is not None and not pd.isna(x) and x > 0.0)
                else (pd.NA if (x is None or pd.isna(x)) else 0),
                na_action=None,
            ).astype("Int64")
            df["target_dir"] = target

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

        avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss != 0.0, 100.0)
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
