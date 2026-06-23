"""Machine-learning strategy trainer for directional classification.

This module provides the :class:`MLStrategyTrainer` class which trains a
Random Forest classifier to predict whether the 5-day forward return of an
asset is positive, and evaluates it with time-series-aware cross-validation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

logger: logging.Logger = logging.getLogger(__name__)

# Feature columns consumed by the model (must be present after transform).
FEATURE_COLUMNS: Final[Tuple[str, ...]] = (
    "log_return",
    "sma_20",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_width",
    "atr_pct",
    "vol_ratio",
)

TARGET_COLUMN: Final[str] = "target_dir"

_MODELS_DIR: Final[Path] = Path("models")


class MLStrategyTrainer:
    """Train and cross-validate a Random Forest directional classifier.

    The trainer ingests a feature-engineered DataFrame (produced by
    :class:`~features.engineer.FeatureEngineer` with ``add_target=True``),
    performs time-series-safe cross-validation, logs out-of-fold Precision
    and Recall, and saves the final model fitted on the entire training set.

    Attributes:
        n_estimators: Number of trees in the Random Forest.
        n_splits: Number of folds for :class:`~sklearn.model_selection.TimeSeriesSplit`.
        random_state: Seed for reproducibility.
        models_dir: Directory where the trained model artifact is saved.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        n_splits: int = 5,
        random_state: int = 42,
        models_dir: Path | str = _MODELS_DIR,
    ) -> None:
        """Initialize the ML strategy trainer.

        Args:
            n_estimators: Number of trees in the Random Forest (must be >= 1).
            n_splits: Number of TimeSeriesSplit folds (must be >= 2).
            random_state: Random seed for the classifier.
            models_dir: Directory where model artifacts are persisted.

        Raises:
            ValueError: If ``n_estimators`` < 1 or ``n_splits`` < 2.
        """
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        self.n_estimators: int = n_estimators
        self.n_splits: int = n_splits
        self.random_state: int = random_state
        self.models_dir: Path = Path(models_dir)

    def _build_model(self) -> RandomForestClassifier:
        """Instantiate a fresh :class:`RandomForestClassifier`.

        Returns:
            An unfitted :class:`RandomForestClassifier` with the configured
            hyper-parameters.
        """
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=6,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=self.random_state,
            class_weight="balanced",
        )

    @staticmethod
    def _prepare_xy(
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and target vector from a DataFrame.

        Args:
            data: Feature-engineered DataFrame that must include
                ``FEATURE_COLUMNS`` and ``TARGET_COLUMN``.

        Returns:
            A ``(X, y)`` tuple where ``X`` is the feature matrix and ``y``
            is the integer-valued target series.

        Raises:
            ValueError: If any required column is absent.
        """
        missing_feat = [c for c in FEATURE_COLUMNS if c not in data.columns]
        if missing_feat:
            raise ValueError(
                f"DataFrame missing feature columns: {missing_feat}"
            )
        if TARGET_COLUMN not in data.columns:
            raise ValueError(
                f"DataFrame missing target column '{TARGET_COLUMN}'. "
                "Call FeatureEngineer.transform(data, add_target=True)."
            )
        X = data[list(FEATURE_COLUMNS)].astype(float)
        y = data[TARGET_COLUMN].astype(int)
        return X, y

    def cross_validate(
        self, data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Run time-series cross-validation and log fold metrics.

        Uses :class:`~sklearn.model_selection.TimeSeriesSplit` so that each
        validation fold is always *after* the training fold in time, fully
        preventing look-ahead bias.

        Args:
            data: Feature-engineered training DataFrame (with ``target_dir``).

        Returns:
            A ``(mean_precision, mean_recall)`` tuple averaged across folds.
        """
        X, y = self._prepare_xy(data)
        tss = TimeSeriesSplit(n_splits=self.n_splits)

        precisions: List[float] = []
        recalls: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(tss.split(X), start=1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = self._build_model()
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            p = float(precision_score(y_val, y_pred, zero_division=0))
            r = float(recall_score(y_val, y_pred, zero_division=0))
            precisions.append(p)
            recalls.append(r)
            logger.info(
                "Fold %d/%d  |  val_size=%d  |  Precision=%.4f  Recall=%.4f",
                fold,
                self.n_splits,
                len(val_idx),
                p,
                r,
            )

        mean_precision = float(np.mean(precisions))
        mean_recall = float(np.mean(recalls))
        logger.info(
            "Out-of-fold CV results  |  Mean Precision=%.4f  Mean Recall=%.4f",
            mean_precision,
            mean_recall,
        )
        return mean_precision, mean_recall

    def train_and_save(
        self, data: pd.DataFrame, model_name: str = "rf_classifier.joblib"
    ) -> Path:
        """Fit the model on the full training set and persist it to disk.

        Args:
            data: Feature-engineered training DataFrame (with ``target_dir``).
            model_name: Filename (no path) for the saved model artifact.

        Returns:
            The :class:`~pathlib.Path` to the saved ``.joblib`` file.
        """
        X, y = self._prepare_xy(data)

        model = self._build_model()
        model.fit(X, y)
        logger.info(
            "Model fitted on %d training samples (%d features).",
            len(X),
            len(FEATURE_COLUMNS),
        )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.models_dir / model_name
        joblib.dump(model, out_path)
        logger.info("Model saved to %s", out_path)
        return out_path
