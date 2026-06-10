"""Abstract base definitions for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Concrete strategies must implement :meth:`generate_signals`, which maps a
    feature-augmented OHLCV DataFrame to a discrete position signal stream.

    The signal convention used throughout the engine is:

    * ``-1`` -- short position.
    * ``0`` -- flat / in cash.
    * ``1`` -- long position.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate a discrete position-signal stream from market data.

        Args:
            data: A DataFrame containing price and engineered feature columns,
                indexed chronologically.

        Returns:
            An integer-valued :class:`~pandas.Series` aligned to ``data``'s
            index, with values in ``{-1, 0, 1}``.

        Raises:
            ValueError: If the input data is missing columns required by the
                concrete strategy.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_columns(data: pd.DataFrame, required: tuple[str, ...]) -> None:
        """Validate that required columns are present in ``data``.

        Args:
            data: DataFrame to validate.
            required: Tuple of column names that must be present.

        Raises:
            ValueError: If ``data`` is empty or any required column is missing.
        """
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing columns: {missing}")
