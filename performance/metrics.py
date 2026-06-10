"""Quantitative risk and performance analytics."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

# Number of trading periods per year assumed for annualization (daily bars).
_TRADING_DAYS_PER_YEAR: int = 252


class RiskAnalytics:
    """Compute risk-adjusted performance metrics from an equity curve.

    The analyzer derives standard quantitative statistics from a strategy's
    equity curve, with an optional list of per-trade PnLs used to compute the
    win/loss ratio.

    Attributes:
        risk_free_rate: Annualized risk-free rate used by the Sharpe ratio.
        periods_per_year: Number of return periods per year (252 for daily).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        periods_per_year: int = _TRADING_DAYS_PER_YEAR,
    ) -> None:
        """Initialize the analyzer.

        Args:
            risk_free_rate: Annualized risk-free rate as a decimal (``0.04`` =
                4%). Used to compute the Sharpe ratio.
            periods_per_year: Number of return observations per year, used for
                annualization (must be >= 1).

        Raises:
            ValueError: If ``periods_per_year`` < 1.
        """
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be >= 1")
        self.risk_free_rate: float = float(risk_free_rate)
        self.periods_per_year: int = int(periods_per_year)

    def analyze(
        self,
        equity_curve: pd.Series,
        trade_pnls: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """Compute a suite of performance metrics from an equity curve.

        Args:
            equity_curve: Series of portfolio values indexed chronologically.
                Must contain at least two observations.
            trade_pnls: Optional sequence of realized per-trade PnLs. When
                provided, the win/loss ratio is computed from individual
                trades; otherwise it falls back to the ratio of up-periods to
                down-periods in the daily returns.

        Returns:
            A dictionary with the keys ``total_return``,
            ``annualized_volatility``, ``sharpe_ratio``, ``max_drawdown`` and
            ``win_loss_ratio``. All values are floats.

        Raises:
            ValueError: If ``equity_curve`` has fewer than two observations or
                contains non-positive values.
        """
        if len(equity_curve) < 2:
            raise ValueError("equity_curve must have at least two observations")
        if (equity_curve <= 0).any():
            raise ValueError("equity_curve must be strictly positive")

        returns = equity_curve.pct_change().dropna()

        metrics: Dict[str, float] = {
            "total_return": self._total_return(equity_curve),
            "annualized_volatility": self._annualized_volatility(returns),
            "sharpe_ratio": self._sharpe_ratio(returns),
            "max_drawdown": self._max_drawdown(equity_curve),
            "win_loss_ratio": self._win_loss_ratio(returns, trade_pnls),
        }
        return metrics

    @staticmethod
    def _total_return(equity_curve: pd.Series) -> float:
        """Compute the total return over the full equity curve.

        Args:
            equity_curve: Series of portfolio values.

        Returns:
            The cumulative return as a decimal (``0.25`` = +25%).
        """
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)

    def _annualized_volatility(self, returns: pd.Series) -> float:
        """Compute annualized volatility of periodic returns.

        Args:
            returns: Series of periodic (e.g. daily) simple returns.

        Returns:
            The annualized standard deviation of returns. Returns ``0.0`` when
            there are too few observations to compute a sample std.
        """
        if len(returns) < 2:
            return 0.0
        return float(returns.std(ddof=1) * np.sqrt(self.periods_per_year))

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Compute the annualized Sharpe ratio.

        Args:
            returns: Series of periodic simple returns.

        Returns:
            The annualized Sharpe ratio, or ``0.0`` if volatility is zero or
            there are insufficient observations.
        """
        if len(returns) < 2:
            return 0.0
        std = returns.std(ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0
        periodic_rf = self.risk_free_rate / self.periods_per_year
        excess = returns - periodic_rf
        sharpe = excess.mean() / std * np.sqrt(self.periods_per_year)
        return float(sharpe)

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        """Compute the maximum peak-to-trough drawdown.

        Args:
            equity_curve: Series of portfolio values.

        Returns:
            The maximum drawdown as a negative decimal (e.g. ``-0.30`` for a
            30% drawdown). Returns ``0.0`` if the curve never declines.
        """
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        return float(drawdown.min())

    @staticmethod
    def _win_loss_ratio(
        returns: pd.Series, trade_pnls: Optional[Sequence[float]]
    ) -> float:
        """Compute the win/loss ratio.

        Args:
            returns: Series of periodic returns (used as a fallback).
            trade_pnls: Optional realized per-trade PnLs.

        Returns:
            The ratio of winning to losing observations. If there are no
            losers but at least one winner, returns positive infinity; if there
            are neither, returns ``0.0``.
        """
        if trade_pnls is not None and len(trade_pnls) > 0:
            arr = np.asarray(trade_pnls, dtype=float)
            wins = int((arr > 0).sum())
            losses = int((arr < 0).sum())
        else:
            wins = int((returns > 0).sum())
            losses = int((returns < 0).sum())

        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return float(wins / losses)
