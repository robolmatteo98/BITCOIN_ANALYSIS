from __future__ import annotations
import pandas as pd

def example_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Due transazioni collegano A1-A2-A3 transitivamente.
            ("tx_11", "A1", "N1", 60_000, "2024-01-01T10:00:00Z"),
            ("tx_11", "A2", "N1", 40_000, "2024-01-01T10:00:00Z"),
            ("tx_12", "A2", "N2", 55_000, "2024-01-02T11:30:00Z"),
            ("tx_12", "A3", "N2", 45_000, "2024-01-02T11:30:00Z"),
            # Altri tre cluster indipendenti.
            ("tx_13", "B1", "N3", 30_000, "2024-01-03T09:15:00Z"),
            ("tx_13", "B2", "N3", 35_000, "2024-01-03T09:15:00Z"),
            ("tx_13", "B3", "N3", 35_000, "2024-01-03T09:15:00Z"),
            ("tx_14", "C1", "N4", 70_000, "2024-01-04T14:45:00Z"),
            ("tx_14", "C2", "N4", 30_000, "2024-01-04T14:45:00Z"),
            ("tx_15", "D1", "N5", 50_000, "2024-01-05T16:20:00Z"),
            ("tx_15", "D2", "N5", 50_000, "2024-01-05T16:20:00Z"),
        ],
        columns=["tx_id", "source", "destination", "amount", "timestamp"],
    )