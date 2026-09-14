"""Capital-matched benchmark execution and comparison APIs."""

from trading_research.benchmarks.buy_and_hold import (
    BenchmarkAnalysis,
    BenchmarkAssumptions,
    BuyAndHoldBenchmarkStrategy,
    run_buy_and_hold_benchmark,
)
from trading_research.benchmarks.comparison import (
    BenchmarkComparison,
    calculate_benchmark_comparison,
)

__all__ = [
    "BenchmarkAnalysis",
    "BenchmarkAssumptions",
    "BenchmarkComparison",
    "BuyAndHoldBenchmarkStrategy",
    "calculate_benchmark_comparison",
    "run_buy_and_hold_benchmark",
]
