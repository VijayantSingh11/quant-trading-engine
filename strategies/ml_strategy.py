"""ML-based signal strategy backed by a trained Random Forest classifier.

This module provides :class:`MLSignalStrategy`, a concrete implementation of
:class:`~strategies.base.BaseStrategy` that loads a pre-trained scikit-learn
classifier and converts its predicted class probabilities into discrete
position signals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from strategies.base import BaseStrategy
from strategies.ml_model import FEATURE_COLUMNS

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH: Final[Path] = Path("models") / "rf_classifier.joblib"


class MLSignalStrategy(BaseStrategy):
    """Position-signal generator backed by a pre-trained Random Forest.

    At each timestep the strategy asks the model for the predicted probability
    of a positive forward return.  The probability is compared against
    configurable long and short thresholds to produce a discrete signal:

    * ``1``  (long) if ``P(positive) > long_threshold``
    * ``-1`` (short) if ``P(positive) < short_threshold``
    * ``0``  (flat) otherwise

    Attributes:
        model: The loaded scikit-learn classifier.
        long_threshold: Probability threshold above which a Buy signal is
            emitted.
        short_threshold: Probability threshold below which a Short signal is
            emitted.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = _DEFAULT_MODEL_PATH,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
    ) -> None:
        """Initialize the ML signal strategy by loading a trained model.

        Args:
            model_path: Path to a ``.joblib`` file containing a fitted
                scikit-learn classifier.
            long_threshold: Minimum predicted probability of a positive return
                required to emit a long (``1``) signal. Must be in ``(0, 1)``.
            short_threshold: Maximum predicted probability of a positive return
                below which a short (``-1``) signal is emitted. Must be in
                ``(0, 1)`` and strictly less than ``long_threshold``.

        Raises:
            ValueError: If thresholds are outside ``(0, 1)`` or inconsistent.
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if not 0.0 < short_threshold < long_threshold < 1.0:
            raise ValueError(
                "Require 0 < short_threshold < long_threshold < 1, got "
                f"short_threshold={short_threshold}, "
                f"long_threshold={long_threshold}"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Run MLStrategyTrainer.train_and_save() first."
            )

        self.model: RandomForestClassifier = joblib.load(model_path)
        self.long_threshold: float = long_threshold
        self.short_threshold: float = short_threshold
        logger.info("Loaded ML model from %s", model_path)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate discrete position signals from ML class probabilities.

        For each row the model predicts the probability that the forward return
        is positive (class ``1``). The probability is thresholded into a
        ``{-1, 0, 1}`` signal.

        Args:
            data: A DataFrame containing all :data:`~strategies.ml_model.FEATURE_COLUMNS`,
                indexed chronologically.

        Returns:
            An integer :class:`~pandas.Series` aligned to ``data``'s index with
            values in ``{-1, 0, 1}`` (short, flat, long).

        Raises:
            ValueError: If any required feature column is missing from ``data``.
        """
        self._validate_columns(data, FEATURE_COLUMNS)

        X = data[list(FEATURE_COLUMNS)].astype(float)

        # predict_proba returns [P(class=0), P(class=1)] for each row.
        proba = self.model.predict_proba(X)
        # Index of class "1" in the model's class list.
        pos_class_idx = int(np.where(self.model.classes_ == 1)[0][0])
        prob_positive = proba[:, pos_class_idx]

        signals = np.zeros(len(data), dtype=np.int64)
        signals[prob_positive > self.long_threshold] = 1
        signals[prob_positive < self.short_threshold] = -1

        return pd.Series(signals, index=data.index, name="signal")
