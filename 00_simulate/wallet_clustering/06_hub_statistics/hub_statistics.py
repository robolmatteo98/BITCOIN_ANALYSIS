"""Calcolo delle statistiche utili a individuare gli hub."""

from __future__ import annotations

import pandas as pd


def calculate_address_statistics(inputs: pd.DataFrame) -> pd.DataFrame:
    """Restituisce una riga di statistiche per ogni source."""
    records = []
    for address, group in inputs.groupby("source"):
        tx_ids = set(group["tx_id"])
        counterparties = set()
        co_spending_events = 0

        for _, transaction in inputs[inputs["tx_id"].isin(tx_ids)].groupby("tx_id"):
            sources = set(transaction["source"])
            counterparties.update(sources - {address})
            co_spending_events += len(sources - {address})

        timestamps = pd.to_datetime(group["timestamp"], utc=True)
        records.append(
            {
                "source": address,
                "input_tx_count": len(tx_ids),
                "input_row_count": len(group),
                "unique_counterparty_count": len(counterparties),
                "co_spending_events": co_spending_events,
                "first_seen": timestamps.min().isoformat(),
                "last_seen": timestamps.max().isoformat(),
                "active_days": timestamps.dt.date.nunique(),
            }
        )

    return pd.DataFrame(records).sort_values(
        ["input_tx_count", "unique_counterparty_count", "source"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def calculate_transaction_statistics(inputs: pd.DataFrame) -> pd.DataFrame:
    """Restituisce il numero di input per ogni transazione."""
    return (
        inputs.groupby("tx_id", as_index=False)
        .agg(
            input_count=("source", "nunique"),
            source_count=("source", "size"),
            timestamp=("timestamp", "first"),
        )
        .sort_values(["input_count", "tx_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
