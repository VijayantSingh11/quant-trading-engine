"""Unit tests for MLStrategyTrainer and MLSignalStrategy."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.engineer import FeatureEngineer
from strategies.ml_model import FEATURE_COLUMNS, MLStrategyTrainer
from strategies.ml_strategy import MLSignalStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with ``n`` bars."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1.5, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.5, n),
            "High": close + rng.uniform(0.1, 1.5, n),
            "Low": close - rng.uniform(0.1, 1.5, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=index,
    )


def _make_features(n: int = 300, add_target: bool = True) -> pd.DataFrame:
    """Return engineered features from synthetic data."""
    return FeatureEngineer().transform(_make_frame(n), add_target=add_target)


# ── MLStrategyTrainer ─────────────────────────────────────────────────────────

class TestMLStrategyTrainer:
    """Tests for :class:`MLStrategyTrainer`."""

    def test_invalid_n_estimators_raises(self) -> None:
        with pytest.raises(ValueError):
            MLStrategyTrainer(n_estimators=0)

    def test_invalid_n_splits_raises(self) -> None:
        with pytest.raises(ValueError):
            MLStrategyTrainer(n_splits=1)

    def test_cross_validate_returns_floats(self) -> None:
        """cross_validate must return two floats in [0, 1]."""
        data = _make_features()
        trainer = MLStrategyTrainer(n_estimators=10, n_splits=3)
        prec, rec = trainer.cross_validate(data)
        assert isinstance(prec, float)
        assert isinstance(rec, float)
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0

    def test_cross_validate_missing_target_raises(self) -> None:
        """cross_validate must raise when target_dir column is absent."""
        data = _make_features(add_target=False)
        trainer = MLStrategyTrainer(n_estimators=10, n_splits=3)
        with pytest.raises(ValueError, match="target"):
            trainer.cross_validate(data)

    def test_cross_validate_missing_feature_raises(self) -> None:
        """cross_validate must raise when a feature column is absent."""
        data = _make_features()
        data = data.drop(columns=["rsi_14"])
        trainer = MLStrategyTrainer(n_estimators=10, n_splits=3)
        with pytest.raises(ValueError, match="rsi_14"):
            trainer.cross_validate(data)

    def test_train_and_save_creates_file(self) -> None:
        """train_and_save must create a .joblib file at the expected path."""
        data = _make_features()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MLStrategyTrainer(
                n_estimators=10, n_splits=3, models_dir=tmp
            )
            path = trainer.train_and_save(data, model_name="test_model.joblib")
            assert path.exists()
            assert path.suffix == ".joblib"

    def test_train_and_save_model_is_loadable(self) -> None:
        """The saved model must be loadable with joblib and able to predict."""
        import joblib
        data = _make_features()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MLStrategyTrainer(
                n_estimators=10, n_splits=3, models_dir=tmp
            )
            path = trainer.train_and_save(data, model_name="model.joblib")
            model = joblib.load(path)
            X = data[list(FEATURE_COLUMNS)].astype(float)
            preds = model.predict(X)
            assert set(preds).issubset({0, 1})


# ── MLSignalStrategy ──────────────────────────────────────────────────────────

class TestMLSignalStrategy:
    """Tests for :class:`MLSignalStrategy`."""

    def _trained_model_path(self, tmp_dir: str) -> Path:
        """Train a small model and return its path."""
        data = _make_features()
        trainer = MLStrategyTrainer(
            n_estimators=20, n_splits=3, models_dir=tmp_dir
        )
        return trainer.train_and_save(data, model_name="rf.joblib")

    def test_invalid_thresholds_raise(self) -> None:
        """Inconsistent thresholds must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._trained_model_path(tmp)
            with pytest.raises(ValueError):
                MLSignalStrategy(
                    model_path=path,
                    long_threshold=0.4,
                    short_threshold=0.6,
                )

    def test_missing_model_file_raises(self) -> None:
        """A non-existent model path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MLSignalStrategy(model_path="/no/such/model.joblib")

    def test_signals_are_valid(self) -> None:
        """Generated signals must only contain values in {-1, 0, 1}."""
        data = _make_features(add_target=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._trained_model_path(tmp)
            strategy = MLSignalStrategy(model_path=path)
            signals = strategy.generate_signals(data)
            assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signal_index_matches_data(self) -> None:
        """Signal Series index must equal the input DataFrame index."""
        data = _make_features(add_target=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._trained_model_path(tmp)
            strategy = MLSignalStrategy(model_path=path)
            signals = strategy.generate_signals(data)
            assert signals.index.equals(data.index)

    def test_missing_feature_column_raises(self) -> None:
        """Missing a required feature column must raise ValueError."""
        data = _make_features(add_target=False).drop(columns=["macd"])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._trained_model_path(tmp)
            strategy = MLSignalStrategy(model_path=path)
            with pytest.raises(ValueError):
                strategy.generate_signals(data)

    def test_high_long_threshold_produces_fewer_longs(self) -> None:
        """A very high long_threshold should suppress most long signals."""
        data = _make_features(add_target=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._trained_model_path(tmp)
            conservative = MLSignalStrategy(
                model_path=path, long_threshold=0.99, short_threshold=0.01
            )
            signals = conservative.generate_signals(data)
            # With extreme thresholds, signals should be mostly flat.
            assert (signals == 0).sum() > len(signals) * 0.5
