"""End-to-end pipeline entry point.

Downloads several years of daily data, engineers features, runs the baseline
trend-following strategy through the historical backtester, and prints the
resulting performance metrics as JSON to the console.

Example:
    $ python main.py
    $ python main.py --symbols AAPL MSFT --period 3y
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List

from backtester.engine import BacktestResult, HistoricalBacktester
from data_pipeline.ingest import DataIngester, DataIngestionError
from features.engineer import FeatureEngineer
from performance.metrics import RiskAnalytics
from strategies.baseline import TrendFollowingStrategy

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS: List[str] = ["AAPL", "BTC-USD"]


def run_symbol(
    symbol: str,
    period: str,
    ingester: DataIngester,
    engineer: FeatureEngineer,
    strategy: TrendFollowingStrategy,
    backtester: HistoricalBacktester,
    analytics: RiskAnalytics,
) -> Dict[str, float]:
    """Run the full pipeline for a single symbol and return its metrics.

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


def main() -> None:
    """Parse CLI arguments, run the pipeline, and print metrics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Ticker symbols to backtest (default: AAPL BTC-USD).",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="Lookback period for historical data (default: 5y).",
    )
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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


if __name__ == "__main__":
    main()
