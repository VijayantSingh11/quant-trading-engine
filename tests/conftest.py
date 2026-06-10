"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def ohlcv_frame() -> pd.DataFrame:
    """Return a deterministic synthetic OHLCV DataFrame.

    Returns:
        A 250-row daily OHLCV DataFrame with a gentle upward drift and noise,
        suitable for exercising indicators and the backtester offline.
    """
    rng = np.random.default_rng(42)
    n = 250
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    drift = np.linspace(100.0, 160.0, n)
    noise = rng.normal(0.0, 1.5, n).cumsum()
    close = drift + noise
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0.0, 0.5, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )
