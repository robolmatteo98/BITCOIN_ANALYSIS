"""Scrittura dei report prodotti dal wallet clustering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_split(
    output_dir: Path,
    split: str,
    mapping: pd.DataFrame,
    evidence: pd.DataFrame,
    conflicts: pd.DataFrame | None = None,
) -> None:
    """Scrive mapping, evidenze, conflitti e riepilogo di uno split."""
    # Ogni split ha una directory autonoma per poter confrontare storico e
    # finestra futura senza sovrascrivere i risultati.
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(split_dir / "address_to_wallet.csv", index=False)
    evidence.to_csv(split_dir / "clustering_evidence.csv", index=False)
    if conflicts is not None:
        conflicts.to_json(split_dir / "conflicts.json", orient="records", indent=2)

    summary = {
        "split": split,
        "address_count": len(mapping),
        "wallet_count": int(mapping["wallet_id"].nunique()) if not mapping.empty else 0,
        "evidence_count": len(evidence),
        "conflict_count": len(conflicts) if conflicts is not None else 0,
    }
    (split_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    output_dir: Path,
    database: str | None,
    transaction_ids: list[int],
    history_limit: int,
) -> None:
    """Salva la finestra esatta usata per costruire i wallet.

    Il manifest evita che le pipeline successive ricostruiscano una finestra
    simile ma non identica usando query SQL differenti.
    """
    history_ids = transaction_ids[:history_limit]
    analysis_ids = transaction_ids[history_limit:]
    manifest = {
        "database": database or "from_env",
        "limit": len(transaction_ids),
        "history_limit": history_limit,
        "analysis_limit": len(analysis_ids),
        "transaction_ids": transaction_ids,
        "history_transaction_ids": history_ids,
        "analysis_transaction_ids": analysis_ids,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
