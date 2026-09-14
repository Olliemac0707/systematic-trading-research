"""Exact arithmetic shared by simulated execution and pre-trade controls."""

from trading_research.simulation.costs import (
    calculate_buy_cost,
    calculate_buy_fill_price,
    calculate_proportional_commission,
    calculate_sell_fill_price,
    largest_affordable_quantity,
)

__all__ = [
    "calculate_buy_cost",
    "calculate_buy_fill_price",
    "calculate_proportional_commission",
    "calculate_sell_fill_price",
    "largest_affordable_quantity",
]
