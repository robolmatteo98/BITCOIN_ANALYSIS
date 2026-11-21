from __future__ import annotations

import pandas as pd


def example_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("tx_1", "A", 100_000),
            ("tx_2", "A", 70_000),
            ("tx_2", "C", 30_000),
        ],
        columns=["tx_id", "address", "amount"],
    )


def example_outputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("tx_1", 0, "B", 30_000),
            ("tx_1", 1, "A_change", 69_000),
            ("tx_2", 0, "D", 20_000),
            ("tx_2", 1, "E", 79_000),
        ],
        columns=["tx_id", "output_index", "address", "amount"],
    )
