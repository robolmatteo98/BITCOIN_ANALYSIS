#!/usr/bin/env python3
"""Entry point della pipeline temporale di wallet/entity clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

from change_address.algorithm import find_change_candidates
from data_loader import load_transactions
from entity_clustering import (
    build_entity_clusters,
    complete_mapping_with_observed_addresses,
    extend_entity_clusters,
)
from report_writer import write_manifest, write_split


ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    """Definisce le opzioni configurabili da terminale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", metavar="NOME_DB")
    parser.add_argument("--limit", type=int, default=11000)
    parser.add_argument("--history-limit", type=int, default=10000)
    parser.add_argument("--coinjoin-threshold", type=float, default=0.75)
    parser.add_argument("--change-threshold", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "07_reports" / "wallet_clustering" / "entity_clustering",
    )
    args = parser.parse_args()
    if args.limit < 1 or args.history_limit < 1 or args.history_limit >= args.limit:
        parser.error("servono limit e history-limit positivi, con history-limit < limit")
    return args


def main() -> int:
    """Coordina caricamento, scoring, clustering e scrittura dei report."""
    args = parse_arguments()

    # 1. Carichiamo dalla stessa finestra temporale input, output, script type
    #    e score CoinJoin.
    inputs, outputs, coinjoin, transaction_ids = load_transactions(
        args.database, args.limit
    )

    # 2. Calcoliamo il punteggio di ogni output. Solo i candidati sopra soglia
    #    possono aggiungere un indirizzo al wallet degli input.
    changes = find_change_candidates(
        inputs,
        outputs,
        threshold=args.change_threshold,
    )

    # 3. Separiamo rigorosamente il prefisso storico dalla finestra futura.
    history_ids = set(transaction_ids[:args.history_limit])
    history_inputs = inputs[inputs["tx_id"].isin(history_ids)]
    analysis_inputs = inputs[~inputs["tx_id"].isin(history_ids)]
    history_outputs = outputs[outputs["tx_id"].isin(history_ids)]
    analysis_outputs = outputs[~outputs["tx_id"].isin(history_ids)]
    history_changes = changes[changes["tx_id"].isin(history_ids)]
    analysis_changes = changes[~changes["tx_id"].isin(history_ids)]
    history_coinjoin = coinjoin[coinjoin["transaction_id"].isin(history_ids)]
    analysis_coinjoin = coinjoin[~coinjoin["transaction_id"].isin(history_ids)]

    # 4. Lo storico crea gli ID seriali iniziali dei wallet.
    history_mapping, history_evidence = build_entity_clusters(
        history_inputs,
        history_changes,
        history_coinjoin,
        coinjoin_threshold=args.coinjoin_threshold,
        change_threshold=args.change_threshold,
    )
    history_mapping = complete_mapping_with_observed_addresses(
        history_mapping, history_inputs, history_outputs
    )

    # 5. L'analisi estende il mapping senza fondere retroattivamente wallet
    #    storici distinti quando una transazione e ambigua.
    analysis_mapping, analysis_evidence, conflicts = extend_entity_clusters(
        analysis_inputs,
        analysis_changes,
        analysis_coinjoin,
        history_mapping,
        coinjoin_threshold=args.coinjoin_threshold,
        change_threshold=args.change_threshold,
    )
    analysis_mapping = complete_mapping_with_observed_addresses(
        analysis_mapping, analysis_inputs, analysis_outputs
    )

    # 6. Salviamo i risultati in directory separate e riproducibili.
    write_split(args.output_dir, "history", history_mapping, history_evidence)
    write_split(
        args.output_dir,
        "analysis",
        analysis_mapping,
        analysis_evidence,
        conflicts,
    )
    write_manifest(
        args.output_dir,
        args.database,
        transaction_ids,
        args.history_limit,
    )

    print(
        f"History: {len(history_mapping)} address, "
        f"{history_mapping['wallet_id'].nunique() if not history_mapping.empty else 0} wallet"
    )
    print(
        f"Analysis: {len(analysis_mapping)} address, "
        f"{analysis_mapping['wallet_id'].nunique() if not analysis_mapping.empty else 0} wallet"
    )
    print(f"Conflitti: {len(conflicts)}")
    print(f"Risultati salvati in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
