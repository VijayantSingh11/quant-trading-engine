"""Historical market-data ingestion utilities.

This module provides the :class:`DataIngester` class which fetches historical
daily OHLCV data from Yahoo Finance via :mod:`yfinance`, retries transient
failures with exponential backoff, and persists the raw results to local
Apache Parquet files for fast subsequent I/O.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final, List, Optional, Union

import pandas as pd
import yfinance as yf

logger: logging.Logger = logging.getLogger(__name__)

_OHLCV_COLUMNS: Final[List[str]] = ["Open", "High", "Low", "Close", "Volume"]


class DataIngestionError(RuntimeError):
    """Raised when market data cannot be ingested after all retries."""


class DataIngester:
    """Fetch and persist historical daily OHLCV data.

    The ingester wraps :func:`yfinance.download` with strict validation,
    exponential-backoff retry handling for transient/rate-limit errors, and
    transparent caching of raw data as Apache Parquet files on local disk.

    Attributes:
        storage_dir: Directory in which Parquet files are stored.
        max_retries: Maximum number of download attempts before giving up.
        backoff_base: Base (in seconds) for the exponential backoff delay.
        backoff_cap: Maximum delay (in seconds) between retries.
    """

    def __init__(
        self,
        storage_dir: Union[str, Path] = "data",
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 60.0,
    ) -> None:
        """Initialize the ingester.

        Args:
            storage_dir: Directory used to store Parquet files. Created if it
                does not already exist.
            max_retries: Maximum number of download attempts (must be >= 1).
            backoff_base: Base delay in seconds for exponential backoff. The
                delay for attempt ``n`` (0-indexed) is
                ``min(backoff_base * 2 ** n, backoff_cap)``.
            backoff_cap: Upper bound, in seconds, on any single backoff delay.

        Raises:
            ValueError: If ``max_retries`` < 1 or any delay parameter is
                negative.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if backoff_base < 0 or backoff_cap < 0:
            raise ValueError("backoff parameters must be non-negative")

        self.storage_dir: Path = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries: int = max_retries
        self.backoff_base: float = backoff_base
        self.backoff_cap: float = backoff_cap

    def _parquet_path(self, symbol: str, period: str, interval: str) -> Path:
        """Return the on-disk Parquet path for a given request.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            period: Lookback period string (e.g. ``"5y"``).
            interval: Sampling interval (e.g. ``"1d"``).

        Returns:
            Absolute :class:`~pathlib.Path` to the cache file.
        """
        safe_symbol = symbol.replace("/", "_").replace("\\", "_").upper()
        filename = f"{safe_symbol}_{period}_{interval}.parquet"
        return self.storage_dir / filename

    def _backoff_delay(self, attempt: int) -> float:
        """Compute the exponential-backoff delay for a retry attempt.

        Args:
            attempt: Zero-indexed attempt number.

        Returns:
            Delay in seconds, capped at ``backoff_cap``.
        """
        return min(self.backoff_base * (2 ** attempt), self.backoff_cap)

    @staticmethod
    def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
        """Flatten and standardize the columns returned by yfinance.

        ``yfinance`` may return a :class:`~pandas.MultiIndex` on the columns
        when a single ticker is requested. This helper flattens that to the
        canonical OHLCV column names.

        Args:
            data: Raw DataFrame returned by :func:`yfinance.download`.

        Returns:
            A DataFrame with single-level OHLCV column names.
        """
        if isinstance(data.columns, pd.MultiIndex):
            # Drop the ticker level, keeping the OHLCV field level.
            data = data.copy()
            data.columns = data.columns.get_level_values(0)
        return data

    def fetch(
        self,
        symbol: str,
        period: str = "5y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data, using the local cache when possible.

        Args:
            symbol: Ticker symbol to download (e.g. ``"AAPL"`` or
                ``"BTC-USD"``).
            period: Amount of history to request (e.g. ``"5y"``, ``"1y"``,
                ``"max"``).
            interval: Bar interval (e.g. ``"1d"``, ``"1h"``).
            force_refresh: If ``True``, ignore any cached Parquet file and
                re-download from the network.

        Returns:
            A DataFrame indexed by date with at least the columns
            ``Open``, ``High``, ``Low``, ``Close`` and ``Volume``.

        Raises:
            ValueError: If ``symbol`` is empty.
            DataIngestionError: If no data could be retrieved after all
                retries, or the returned payload is empty/malformed.
        """
        if not symbol or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        path = self._parquet_path(symbol, period, interval)
        if path.exists() and not force_refresh:
            logger.info("Loading cached data for %s from %s", symbol, path)
            return self.load(path)

        data = self._download_with_retries(symbol, period, interval)
        data = self._normalize_columns(data)
        self._validate(symbol, data)
        self.save(data, path)
        logger.info("Fetched %d rows for %s", len(data), symbol)
        return data

    def _download_with_retries(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """Download data, retrying transient failures with backoff.

        Args:
            symbol: Ticker symbol to download.
            period: Lookback period string.
            interval: Bar interval string.

        Returns:
            The raw DataFrame returned by :func:`yfinance.download`.

        Raises:
            DataIngestionError: If every attempt fails or returns no data.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if data is not None and not data.empty:
                    return data
                last_error = DataIngestionError(
                    f"Empty payload returned for {symbol!r}"
                )
            except Exception as exc:  # noqa: BLE001 - retried & re-raised below
                last_error = exc
                logger.warning(
                    "Download attempt %d/%d for %s failed: %s",
                    attempt + 1,
                    self.max_retries,
                    symbol,
                    exc,
                )

            if attempt < self.max_retries - 1:
                delay = self._backoff_delay(attempt)
                logger.info("Retrying %s in %.1fs", symbol, delay)
                time.sleep(delay)

        raise DataIngestionError(
            f"Failed to fetch data for {symbol!r} after {self.max_retries} "
            f"attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _validate(symbol: str, data: pd.DataFrame) -> None:
        """Validate that a DataFrame contains the required OHLCV columns.

        Args:
            symbol: Ticker symbol (used only for error messages).
            data: DataFrame to validate.

        Raises:
            DataIngestionError: If the DataFrame is empty or is missing any
                required OHLCV column.
        """
        if data.empty:
            raise DataIngestionError(f"No rows returned for {symbol!r}")
        missing = [c for c in _OHLCV_COLUMNS if c not in data.columns]
        if missing:
            raise DataIngestionError(
                f"Data for {symbol!r} missing required columns: {missing}"
            )

    def save(self, data: pd.DataFrame, path: Union[str, Path]) -> Path:
        """Persist a DataFrame to Apache Parquet.

        Args:
            data: DataFrame to persist.
            path: Destination Parquet path.

        Returns:
            The :class:`~pathlib.Path` the data was written to.

        Raises:
            DataIngestionError: If the file could not be written.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(path, engine="pyarrow", compression="snappy")
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            raise DataIngestionError(
                f"Failed to write Parquet file {path}: {exc}"
            ) from exc
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> pd.DataFrame:
        """Load a DataFrame previously stored as Parquet.

        Args:
            path: Source Parquet path.

        Returns:
            The loaded DataFrame.

        Raises:
            DataIngestionError: If the file does not exist or cannot be read.
        """
        path = Path(path)
        if not path.exists():
            raise DataIngestionError(f"Parquet file does not exist: {path}")
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            raise DataIngestionError(
                f"Failed to read Parquet file {path}: {exc}"
            ) from exc
