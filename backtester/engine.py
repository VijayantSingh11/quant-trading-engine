"""Chronological, bias-free historical backtesting engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single completed round-trip trade.

    Attributes:
        direction: ``1`` for a long round-trip, ``-1`` for a short round-trip.
        entry_time: Timestamp at which the position was opened.
        exit_time: Timestamp at which the position was closed.
        entry_price: Execution price at entry.
        exit_price: Execution price at exit.
        pnl: Realized profit/loss in account currency, net of fees.
    """

    direction: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float


@dataclass
class BacktestResult:
    """Container for the outputs of a backtest run.

    Attributes:
        equity_curve: Mark-to-market portfolio value indexed by date.
        positions: Signed position direction held on each date (``-1/0/1``).
        trades: List of completed round-trip :class:`Trade` objects.
        initial_capital: Starting capital used for the run.
    """

    equity_curve: pd.Series
    positions: pd.Series
    trades: List[Trade] = field(default_factory=list)
    initial_capital: float = 0.0


class HistoricalBacktester:
    """Event-driven backtester that simulates a strategy chronologically.

    The engine iterates over time in strict chronological order. Signals are
    lagged by one bar before execution so that a signal computed from the close
    of day ``t`` is only acted upon at day ``t + 1`` -- this completely
    eliminates look-ahead bias.

    Position sizing invests a fixed fraction of the *current* total portfolio
    value on each new position, and a transaction fee is charged on the traded
    notional of every entry and exit to approximate slippage and brokerage.

    Attributes:
        initial_capital: Starting account value.
        fee_pct: Proportional transaction fee charged per trade (per side).
        position_size: Fraction of current equity allocated per position.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.001,
        position_size: float = 1.0,
    ) -> None:
        """Initialize the backtester.

        Args:
            initial_capital: Starting capital (must be > 0). Defaults to
                ``100_000``.
            fee_pct: Proportional fee per trade side (e.g. ``0.001`` = 0.1%).
                Must be in ``[0, 1)``.
            position_size: Fraction of current portfolio value to deploy on
                each new position. Must be in ``(0, 1]``.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0.0 <= fee_pct < 1.0:
            raise ValueError("fee_pct must be in [0, 1)")
        if not 0.0 < position_size <= 1.0:
            raise ValueError("position_size must be in (0, 1]")

        self.initial_capital: float = float(initial_capital)
        self.fee_pct: float = float(fee_pct)
        self.position_size: float = float(position_size)

    def _position_units(self, equity: float, price: float, direction: int) -> float:
        """Compute the signed number of units for a new position.

        Args:
            equity: Current total portfolio value available to deploy.
            price: Execution price per unit.
            direction: Desired position direction (``-1`` or ``1``).

        Returns:
            The signed unit quantity to hold.
        """
        notional = self.position_size * equity
        return direction * notional / price

    def run(self, data: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """Run the backtest over aligned price data and signals.

        Args:
            data: A DataFrame containing a ``Close`` column, indexed
                chronologically.
            signals: An integer position-signal series (values in
                ``{-1, 0, 1}``) aligned to ``data``'s index.

        Returns:
            A :class:`BacktestResult` with the equity curve, per-bar positions,
            and the list of completed trades.

        Raises:
            ValueError: If ``data`` lacks a ``Close`` column, is empty, or its
                index is not aligned with ``signals``.
        """
        if "Close" not in data.columns:
            raise ValueError("data must contain a 'Close' column")
        if data.empty:
            raise ValueError("data is empty")
        if not data.index.equals(signals.index):
            raise ValueError("data and signals must share the same index")

        # Lag signals by one bar to act on the *next* bar (no look-ahead).
        target_dir = signals.shift(1).fillna(0).astype(int)

        close = data["Close"].to_numpy(dtype=float)
        targets = target_dir.to_numpy(dtype=int)
        index = data.index

        cash: float = self.initial_capital
        units: float = 0.0
        equity_pre_entry: float = self.initial_capital
        entry_time: pd.Timestamp = index[0]
        entry_price: float = close[0]
        entry_dir: int = 0

        equity_curve: List[float] = []
        positions: List[int] = []
        trades: List[Trade] = []

        for i in range(len(close)):
            price = close[i]
            ts = index[i]
            desired = int(targets[i])
            current = int(np.sign(units))

            if desired != current:
                # Close any existing position first.
                if units != 0.0:
                    cash += units * price
                    cash -= abs(units) * price * self.fee_pct
                    equity_post_exit = cash
                    trades.append(
                        Trade(
                            direction=entry_dir,
                            entry_time=entry_time,
                            exit_time=ts,
                            entry_price=entry_price,
                            exit_price=price,
                            pnl=equity_post_exit - equity_pre_entry,
                        )
                    )
                    units = 0.0

                # Open the new position, if any.
                if desired != 0:
                    equity_pre_entry = cash
                    new_units = self._position_units(cash, price, desired)
                    cash -= new_units * price
                    cash -= abs(new_units) * price * self.fee_pct
                    units = new_units
                    entry_time = ts
                    entry_price = price
                    entry_dir = desired

            equity_curve.append(cash + units * price)
            positions.append(int(np.sign(units)))

        # Force-close any open position at the final bar for clean accounting.
        if units != 0.0:
            final_price = close[-1]
            cash += units * final_price
            cash -= abs(units) * final_price * self.fee_pct
            trades.append(
                Trade(
                    direction=entry_dir,
                    entry_time=entry_time,
                    exit_time=index[-1],
                    entry_price=entry_price,
                    exit_price=final_price,
                    pnl=cash - equity_pre_entry,
                )
            )
            equity_curve[-1] = cash
            positions[-1] = 0
            units = 0.0

        logger.info(
            "Backtest complete: %d bars, %d trades, final equity %.2f",
            len(close),
            len(trades),
            equity_curve[-1] if equity_curve else self.initial_capital,
        )

        return BacktestResult(
            equity_curve=pd.Series(equity_curve, index=index, name="equity"),
            positions=pd.Series(positions, index=index, name="position"),
            trades=trades,
            initial_capital=self.initial_capital,
        )
