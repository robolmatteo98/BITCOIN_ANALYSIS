#!/usr/bin/env python3
"""Stampa le statistiche degli indirizzi e delle transazioni statiche."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_co_spending"))

from example_data import example_inputs
from hub_statistics import calculate_address_statistics, calculate_transaction_statistics


def main() -> int:
    inputs = example_inputs()
    address_statistics = calculate_address_statistics(inputs)
    transaction_statistics = calculate_transaction_statistics(inputs)

    print("=== STATISTICHE SOURCE ===")
    print(address_statistics.to_string(index=False))
    print("\n=== STATISTICHE TRANSAZIONI ===")
    print(transaction_statistics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
