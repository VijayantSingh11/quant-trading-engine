"""End-to-end pipeline entry point.

Downloads 5 years of daily BTC-USD data, engineers features, trains a Random
Forest directional classifier on the first 4 years, then backtests both the ML
strategy and a Buy-and-Hold benchmark on the final year (walk-forward test).
Sharpe ratio and Max Drawdown are printed for both strategies.

The original multi-symbol baseline pipeline is retained via the ``--baseline``
flag for backwards compatibility.

Example::

    $ python main.py                          # ML pipeline (BTC-USD)
    $ python main.py --baseline               # original baseline pipeline
    $ python main.py --baseline --symbols AAPL MSFT --period 3y
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from backtester.engine import BacktestResult, HistoricalBacktester
from data_pipeline.ingest import DataIngester, DataIngestionError
from features.engineer import FeatureEngineer
from performance.metrics import RiskAnalytics
from strategies.baseline import TrendFollowingStrategy
from strategies.ml_model import MLStrategyTrainer
from strategies.ml_strategy import MLSignalStrategy

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS: List[str] = ["AAPL", "BTC-USD"]
ML_SYMBOL: str = "BTC-USD"
TRAIN_YEARS: int = 4
TOTAL_YEARS: int = 5
MODEL_PATH: str = "models/rf_classifier.joblib"


# ── Baseline (original) pipeline ─────────────────────────────────────────────

def run_symbol(
    symbol: str,
    period: str,
    ingester: DataIngester,
    engineer: FeatureEngineer,
    strategy: TrendFollowingStrategy,
    backtester: HistoricalBacktester,
    analytics: RiskAnalytics,
) -> Dict[str, float]:
    """Run the full baseline pipeline for a single symbol and return metrics.

    Args:
        symbol: Ticker symbol to process.
        period: Lookback period string passed to the ingester (e.g. ``"5y"``).
        ingester: Configured :class:`DataIngester` instance.
        engineer: Configured :class:`FeatureEngineer` instance.
        strategy: Strategy used to generate signals.
        backtester: Configured :class:`HistoricalBacktester` instance.
        analytics: Configured :class:`RiskAnalytics` instance.

    Returns:
        A dictionary of performance metrics for ``symbol``.

    Raises:
        DataIngestionError: If data for ``symbol`` cannot be retrieved.
    """
    raw = ingester.fetch(symbol, period=period, interval="1d")
    features = engineer.transform(raw)
    signals = strategy.generate_signals(features)
    result: BacktestResult = backtester.run(features, signals)
    trade_pnls = [t.pnl for t in result.trades]
    metrics = analytics.analyze(result.equity_curve, trade_pnls=trade_pnls)
    metrics["final_equity"] = float(result.equity_curve.iloc[-1])
    metrics["num_trades"] = float(len(result.trades))
    return metrics


def run_baseline(args: argparse.Namespace) -> None:
    """Execute the original baseline pipeline and print metrics as JSON.

    Args:
        args: Parsed CLI arguments.
    """
    ingester = DataIngester()
    engineer = FeatureEngineer()
    strategy = TrendFollowingStrategy()
    backtester = HistoricalBacktester(
        initial_capital=args.capital, fee_pct=args.fee
    )
    analytics = RiskAnalytics(risk_free_rate=0.04)

    all_metrics: Dict[str, Dict[str, float]] = {}
    for symbol in args.symbols:
        try:
            all_metrics[symbol] = run_symbol(
                symbol,
                args.period,
                ingester,
                engineer,
                strategy,
                backtester,
                analytics,
            )
        except DataIngestionError as exc:
            logger.error("Skipping %s: %s", symbol, exc)
            all_metrics[symbol] = {"error": str(exc)}  # type: ignore[dict-item]

    print(json.dumps(all_metrics, indent=2, default=str))


# ── ML pipeline ──────────────────────────────────────────────────────────────

def _chronological_split(
    data: pd.DataFrame, train_years: int, total_years: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train and test sets chronologically.

    Args:
        data: Time-indexed DataFrame to split.
        train_years: Number of years to include in the training set.
        total_years: Total number of years in ``data``.

    Returns:
        A ``(train, test)`` tuple of DataFrames.
    """
    split_frac = train_years / total_years
    split_idx = int(len(data) * split_frac)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    logger.info(
        "Chronological split: train=%d rows (%s to %s), test=%d rows (%s to %s)",
        len(train),
        train.index[0].date(),
        train.index[-1].date(),
        len(test),
        test.index[0].date(),
        test.index[-1].date(),
    )
    return train, test


def _buy_and_hold_metrics(
    test_data: pd.DataFrame,
    initial_capital: float,
    analytics: RiskAnalytics,
) -> Dict[str, float]:
    """Compute Sharpe and Max Drawdown for a simple Buy-and-Hold benchmark.

    Args:
        test_data: Feature-engineered test DataFrame with a ``Close`` column.
        initial_capital: Starting capital for the hypothetical portfolio.
        analytics: Configured :class:`RiskAnalytics` instance.

    Returns:
        A dictionary with ``sharpe_ratio``, ``max_drawdown`` and
        ``total_return`` for the benchmark.
    """
    close = test_data["Close"]
    equity = initial_capital * close / close.iloc[0]
    metrics = analytics.analyze(equity)
    return {
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "total_return": metrics["total_return"],
    }


def run_ml_pipeline(args: argparse.Namespace) -> None:
    """Execute the ML training + walk-forward backtest pipeline.

    Steps:
        1. Ingest 5 years of daily BTC-USD data.
        2. Engineer features (with ``add_target=True`` for the training set).
        3. Chronologically split: 4 years train, 1 year test.
        4. Cross-validate and train the Random Forest; log Precision & Recall.
        5. Backtest :class:`MLSignalStrategy` on the test set.
        6. Compute Buy-and-Hold benchmark metrics.
        7. Print a comparison of Sharpe ratio and Max Drawdown.

    Args:
        args: Parsed CLI arguments.
    """
    ingester = DataIngester()
    engineer = FeatureEngineer()
    analytics = RiskAnalytics(risk_free_rate=0.04)
    backtester = HistoricalBacktester(
        initial_capital=args.capital, fee_pct=args.fee
    )

    # ── 1. Ingest ────────────────────────────────────────────────────────────
    logger.info("Ingesting 5 years of daily data for %s …", ML_SYMBOL)
    raw = ingester.fetch(ML_SYMBOL, period="5y", interval="1d")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    # Compute all features INCLUDING the target on the full dataset, then split.
    # The target is only used during training; no signal in the test set
    # depends on future prices because the backtester re-uses only feature
    # columns (not target_dir) when generating signals.
    full_features = engineer.transform(raw, add_target=True)

    # ── 3. Chronological train / test split ──────────────────────────────────
    train_data, test_data = _chronological_split(
        full_features, train_years=TRAIN_YEARS, total_years=TOTAL_YEARS
    )

    # Drop rows from the training tail where target_dir is NaN (the last
    # `forward_return_window` bars have no valid forward return).
    train_data = train_data.dropna(subset=["target_dir"])

    # ── 4. Train the Random Forest ───────────────────────────────────────────
    trainer = MLStrategyTrainer(n_estimators=200, n_splits=5, random_state=42)
    logger.info("Running TimeSeriesSplit cross-validation …")
    mean_precision, mean_recall = trainer.cross_validate(train_data)
    print(
        f"\n{'─'*55}\n"
        f"  Out-of-fold CV  |  Precision={mean_precision:.4f}  "
        f"Recall={mean_recall:.4f}\n"
        f"{'─'*55}"
    )

    model_path = trainer.train_and_save(
        train_data, model_name="rf_classifier.joblib"
    )

    # ── 5. Walk-forward backtest ─────────────────────────────────────────────
    ml_strategy = MLSignalStrategy(
        model_path=model_path,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )
    ml_signals = ml_strategy.generate_signals(test_data)
    ml_result: BacktestResult = backtester.run(test_data, ml_signals)
    ml_trade_pnls = [t.pnl for t in ml_result.trades]
    ml_metrics = analytics.analyze(ml_result.equity_curve, trade_pnls=ml_trade_pnls)

    # ── 6. Buy-and-Hold benchmark ────────────────────────────────────────────
    bah_metrics = _buy_and_hold_metrics(test_data, args.capital, analytics)

    # ── 7. Print comparison ──────────────────────────────────────────────────
    width = 55
    print(f"\n{'═'*width}")
    print(f"  Walk-Forward Test Results  ({ML_SYMBOL}, final year)")
    print(f"{'═'*width}")
    print(f"  {'Metric':<25} {'ML Strategy':>12} {'Buy & Hold':>12}")
    print(f"  {'─'*25} {'─'*12} {'─'*12}")
    for key, label in [
        ("sharpe_ratio", "Sharpe Ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("total_return", "Total Return"),
    ]:
        ml_val = ml_metrics[key]
        bah_val = bah_metrics[key]
        print(f"  {label:<25} {ml_val:>12.4f} {bah_val:>12.4f}")
    print(f"{'═'*width}\n")
    print(f"  ML trades executed : {len(ml_result.trades)}")
    print(f"  Final ML equity    : ${ml_result.equity_curve.iloc[-1]:,.2f}")
    print(f"{'═'*width}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the original baseline (EMA crossover) pipeline instead of ML.",
    )
    # Baseline-only flags
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Ticker symbols for the baseline pipeline (default: AAPL BTC-USD).",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="Lookback period for the baseline pipeline (default: 5y).",
    )
    # Shared flags
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital (default: 100000).",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="Transaction fee fraction per trade side (default: 0.001).",
    )
    # ML-specific flags
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=0.55,
        dest="long_threshold",
        help="Probability threshold for a long signal (default: 0.55).",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=0.45,
        dest="short_threshold",
        help="Probability threshold for a short signal (default: 0.45).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.baseline:
        run_baseline(args)
    else:
        run_ml_pipeline(args)


if __name__ == "__main__":
    main()
