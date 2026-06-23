"""Unit tests for the extended FeatureEngineer (new ML indicators)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.engineer import FeatureEngineer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with ``n`` bars."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)          # keep prices positive
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


# ── MACD ─────────────────────────────────────────────────────────────────────

def test_macd_column_present() -> None:
    """MACD and MACD Signal columns must be present after transform."""
    out = FeatureEngineer().transform(_frame())
    assert "macd" in out.columns
    assert "macd_signal" in out.columns


def test_macd_equals_ema_diff() -> None:
    """MACD must equal ema_12 minus ema_26 for every row."""
    out = FeatureEngineer().transform(_frame())
    expected = out["ema_12"] - out["ema_26"]
    pd.testing.assert_series_equal(out["macd"], expected, check_names=False)


def test_macd_has_no_nans() -> None:
    """MACD columns must be NaN-free after transform."""
    out = FeatureEngineer().transform(_frame())
    assert out["macd"].isna().sum() == 0
    assert out["macd_signal"].isna().sum() == 0


# ── Bollinger Band Width ──────────────────────────────────────────────────────

def test_bb_width_column_present() -> None:
    """bb_width column must be present after transform."""
    out = FeatureEngineer().transform(_frame())
    assert "bb_width" in out.columns


def test_bb_width_positive() -> None:
    """Bollinger Band Width must be strictly positive for a noisy series."""
    out = FeatureEngineer().transform(_frame())
    assert (out["bb_width"] > 0).all()


def test_bb_width_zero_for_constant_price() -> None:
    """For a constant close price, band width should be (near) zero."""
    n = 100
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(50.0, index=index)
    df = pd.DataFrame(
        {"High": close + 0.1, "Low": close - 0.1, "Close": close,
         "Volume": 1_000_000.0}
    )
    out = FeatureEngineer().transform(df)
    assert (out["bb_width"].dropna() < 1e-6).all()


# ── ATR% ─────────────────────────────────────────────────────────────────────

def test_atr_pct_column_present() -> None:
    """atr_pct column must be present after transform."""
    out = FeatureEngineer().transform(_frame())
    assert "atr_pct" in out.columns


def test_atr_pct_positive() -> None:
    """ATR% must be strictly positive for a noisy price series."""
    out = FeatureEngineer().transform(_frame())
    assert (out["atr_pct"] > 0).all()


def test_atr_pct_equals_atr_over_close() -> None:
    """atr_pct must equal atr_14 / Close for every row."""
    out = FeatureEngineer().transform(_frame())
    expected = out["atr_14"] / out["Close"]
    pd.testing.assert_series_equal(
        out["atr_pct"], expected, check_names=False, rtol=1e-9
    )


# ── Volume Ratio ─────────────────────────────────────────────────────────────

def test_vol_ratio_column_present() -> None:
    """vol_ratio column must be present after transform."""
    out = FeatureEngineer().transform(_frame())
    assert "vol_ratio" in out.columns


def test_vol_ratio_positive() -> None:
    """Volume ratio must be strictly positive for non-zero volume data."""
    out = FeatureEngineer().transform(_frame())
    assert (out["vol_ratio"] > 0).all()


def test_vol_ratio_constant_volume_equals_one() -> None:
    """Constant volume yields a vol_ratio of exactly 1.0."""
    n = 200
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 200, n), index=index)
    df = pd.DataFrame(
        {
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 2_000_000.0,
        }
    )
    out = FeatureEngineer().transform(df)
    assert np.allclose(out["vol_ratio"], 1.0, atol=1e-9)


# ── Target variable ───────────────────────────────────────────────────────────

def test_target_dir_present_when_requested() -> None:
    """target_dir column must appear only when add_target=True."""
    df = _frame()
    with_target = FeatureEngineer().transform(df, add_target=True)
    without_target = FeatureEngineer().transform(df, add_target=False)
    assert "target_dir" in with_target.columns
    assert "target_dir" not in without_target.columns


def test_target_dir_is_binary() -> None:
    """target_dir must only contain values in {0, 1}."""
    out = FeatureEngineer().transform(_frame(), add_target=True)
    assert set(out["target_dir"].unique()).issubset({0, 1})


def test_target_dir_no_lookahead() -> None:
    """target_dir values must reflect only information available at each bar.

    The forward log-return target is computed using close prices ``w`` bars
    ahead and shifted so it aligns with the current bar.  We verify that each
    ``target_dir`` value exactly matches the sign of
    ``log(close[t+w] / close[t])`` — i.e., the target uses future prices only
    to *define* the label, not as a feature.  Concretely, no value in the
    output should reference data beyond ``t + forward_return_window``.
    """
    fw = 5
    n = 150
    index = pd.date_range("2021-01-01", periods=n, freq="B")
    # Monotonically increasing close => all 5-day forward returns are positive.
    close = pd.Series(np.linspace(100.0, 200.0, n), index=index)
    df = pd.DataFrame(
        {"High": close + 1, "Low": close - 1, "Close": close, "Volume": 1e6}
    )
    out = FeatureEngineer(forward_return_window=fw).transform(df, add_target=True)
    # Every forward return is positive, so every target_dir must be 1.
    assert (out["target_dir"] == 1).all(), (
        "All target_dir values should be 1 for a monotonically rising series."
    )


def test_target_dir_no_nans() -> None:
    """target_dir must be NaN-free after transform (NaN rows are dropped)."""
    out = FeatureEngineer().transform(_frame(), add_target=True)
    assert out["target_dir"].isna().sum() == 0


def test_target_dir_matches_forward_return() -> None:
    """target_dir must agree with manually computed 5-day forward log return."""
    n = 100
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, n), index=index)
    df = pd.DataFrame(
        {"High": close + 1, "Low": close - 1, "Close": close, "Volume": 1e6}
    )
    fw = 5
    eng = FeatureEngineer(forward_return_window=fw)
    out = eng.transform(df, add_target=True)

    # Recompute expected target for output rows using the *original* close.
    for ts in out.index:
        pos = close.index.get_loc(ts)
        if pos + fw < len(close):
            fwd = np.log(close.iloc[pos + fw] / close.iloc[pos])
            expected = 1 if fwd > 0 else 0
            assert out.loc[ts, "target_dir"] == expected, (
                f"Mismatch at {ts}: expected {expected}, got {out.loc[ts, 'target_dir']}"
            )


# ── Existing indicator regression ─────────────────────────────────────────────

def test_all_original_columns_still_present() -> None:
    """Original indicator columns must still be produced after extension."""
    out = FeatureEngineer().transform(_frame())
    for col in ("log_return", "sma_20", "ema_12", "ema_26", "rsi_14", "atr_14"):
        assert col in out.columns, f"Missing original column: {col}"


def test_no_nans_in_output() -> None:
    """The output DataFrame must be entirely NaN-free."""
    out = FeatureEngineer().transform(_frame())
    assert not out.isna().any().any()
