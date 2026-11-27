"""Fusione di co-spending e change address in wallet probabili.

Le relazioni sono costruite in ordine temporale e le transazioni CoinJoin
candidate vengono escluse dalle fusioni automatiche. Il risultato comprende
sia il mapping address -> wallet sia le evidenze che hanno prodotto ogni
collegamento.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import pandas as pd


class _UnionFind:
    """Struttura per unire indirizzi senza scansionare tutti i cluster."""
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _addresses(values: Iterable[object]) -> set[str]:
    return {str(value) for value in values if pd.notna(value)}


def _coinjoin_ids(
    coinjoin_scores: pd.DataFrame | None,
    threshold: float,
) -> set[object]:
    if coinjoin_scores is None or coinjoin_scores.empty:
        return set()
    required = {"transaction_id", "coinjoin_score"}
    missing = required - set(coinjoin_scores.columns)
    if missing:
        raise ValueError(
            f"colonne CoinJoin mancanti: {', '.join(sorted(missing))}"
        )
    return set(
        coinjoin_scores.loc[
            pd.to_numeric(coinjoin_scores["coinjoin_score"], errors="raise")
            >= threshold,
            "transaction_id",
        ]
    )


def build_entity_clusters(
    inputs: pd.DataFrame,
    change_candidates: pd.DataFrame | None = None,
    coinjoin_scores: pd.DataFrame | None = None,
    coinjoin_threshold: float = 0.75,
    change_threshold: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unisce indirizzi tramite co-spending e change address.

    ``inputs`` deve contenere ``tx_id`` e ``source`` oppure ``address``.
    ``change_candidates`` deve contenere ``tx_id``, ``address`` e ``score``;
    il detector esistente produce esattamente queste colonne.

    Restituisce:

    - mapping con ``address``, ``wallet_id`` e ``confidence``;
    - evidenze con gli archi usati per le fusioni.
    """
    if "tx_id" not in inputs.columns:
        raise ValueError("colonna input mancante: tx_id")
    input_address_column = (
        "source" if "source" in inputs.columns
        else "address" if "address" in inputs.columns
        else None
    )
    if input_address_column is None:
        raise ValueError("serve la colonna input source oppure address")
    if not 0 <= coinjoin_threshold <= 1 or not 0 <= change_threshold <= 1:
        raise ValueError("le soglie devono essere comprese tra 0 e 1")

    if change_candidates is not None:
        required_change = {"tx_id", "address", "score"}
        missing = required_change - set(change_candidates.columns)
        if missing:
            raise ValueError(
                f"colonne change mancanti: {', '.join(sorted(missing))}"
            )

    # Le CoinJoin candidate non vengono usate per fondere input di utenti
    # diversi: e il principale filtro contro il clustering eccessivo.
    blocked_transactions = _coinjoin_ids(coinjoin_scores, coinjoin_threshold)
    union_find = _UnionFind()
    evidence: list[dict[str, object]] = []
    seen_evidence: set[tuple[str, str, str]] = set()
    address_confidence: dict[str, float] = {}

    def connect(left: str, right: str, reason: str, confidence: float, tx_id: object) -> None:
        pair = tuple(sorted((left, right)))
        key = (*pair, reason)
        union_find.add(left)
        union_find.add(right)
        if left == right or key in seen_evidence:
            return
        seen_evidence.add(key)
        union_find.union(left, right)
        address_confidence[left] = min(address_confidence.get(left, 1.0), confidence)
        address_confidence[right] = min(address_confidence.get(right, 1.0), confidence)
        evidence.append({
            "tx_id": tx_id,
            "address_a": pair[0],
            "address_b": pair[1],
            "reason": reason,
            "confidence": round(float(confidence), 6),
        })

    grouped_inputs = inputs.groupby("tx_id", sort=False)
    change_by_tx: dict[object, pd.DataFrame] = {}
    if change_candidates is not None and not change_candidates.empty:
        candidates = change_candidates.copy()
        candidates["score"] = pd.to_numeric(candidates["score"], errors="raise")
        candidates = candidates[candidates["score"] >= change_threshold]
        for tx_id, group in candidates.groupby("tx_id", sort=False):
            change_by_tx[tx_id] = group

    for tx_id, group in grouped_inputs:
        addresses = sorted(_addresses(group[input_address_column]))
        for address in addresses:
            union_find.add(address)
        if tx_id in blocked_transactions:
            continue

        # Evidenza forte: gli input sono stati firmati insieme. La regola e
        # comunque euristica e non va applicata alle CoinJoin filtrate sopra.
        for left, right in combinations(addresses, 2):
            connect(left, right, "co_spending", 1.0, tx_id)

        # Evidenza più debole: l'output e candidato a essere il resto. Il
        # change non viene trattato come certezza di proprieta comune.
        for row in change_by_tx.get(tx_id, pd.DataFrame()).itertuples(index=False):
            change_address = str(row.address)
            for source in addresses:
                connect(source, change_address, "change_address", float(row.score), tx_id)

    all_addresses = set(union_find.parent)
    roots = sorted({union_find.find(address) for address in all_addresses})
    root_to_wallet_id = {root: index for index, root in enumerate(roots, start=1)}
    mapping_rows = []
    for address in sorted(all_addresses):
        root = union_find.find(address)
        mapping_rows.append({
            "address": address,
            "wallet_id": root_to_wallet_id[root],
            "confidence": round(address_confidence.get(address, 1.0), 6),
        })

    mapping = pd.DataFrame(mapping_rows, columns=["address", "wallet_id", "confidence"])
    evidence_frame = pd.DataFrame(
        evidence,
        columns=["tx_id", "address_a", "address_b", "reason", "confidence"],
    )
    return mapping, evidence_frame


def extend_entity_clusters(
    inputs: pd.DataFrame,
    change_candidates: pd.DataFrame | None,
    coinjoin_scores: pd.DataFrame | None,
    initial_mapping: pd.DataFrame,
    coinjoin_threshold: float = 0.75,
    change_threshold: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estende un mapping storico senza fondere wallet gia esistenti.

    Gli indirizzi nuovi possono essere assegnati a un wallet noto quando una
    transazione contiene un solo wallet storico. Se contiene piu wallet noti,
    il caso viene registrato come conflitto e i wallet storici restano separati.
    """
    required = {"address", "wallet_id"}
    missing = required - set(initial_mapping.columns)
    if missing:
        raise ValueError(
            f"colonne mapping mancanti: {', '.join(sorted(missing))}"
        )
    if "tx_id" not in inputs.columns:
        raise ValueError("colonna input mancante: tx_id")
    input_column = "source" if "source" in inputs.columns else "address"
    if input_column not in inputs.columns:
        raise ValueError("serve la colonna input source oppure address")

    blocked = _coinjoin_ids(coinjoin_scores, coinjoin_threshold)
    raw_mapping = {
        str(row.address): row.wallet_id
        for row in initial_mapping.itertuples(index=False)
    }
    # Gli ID prodotti dallo storico sono già stabili. Non li ricompattiamo e
    # non li riordiniamo: farlo cambierebbe il wallet associato a ogni address
    # quando la finestra analysis viene elaborata.
    current = {
        address: int(wallet_id)
        for address, wallet_id in raw_mapping.items()
    }
    next_id = max(current.values(), default=0) + 1
    evidence: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []

    candidates_by_tx: dict[object, pd.DataFrame] = {}
    if change_candidates is not None and not change_candidates.empty:
        candidates = change_candidates.copy()
        candidates["score"] = pd.to_numeric(candidates["score"], errors="raise")
        candidates = candidates[candidates["score"] >= change_threshold]
        for tx_id, group in candidates.groupby("tx_id", sort=False):
            candidates_by_tx[tx_id] = group

    def new_wallet() -> int:
        nonlocal next_id
        wallet_id = next_id
        next_id += 1
        return wallet_id

    # Manteniamo gli ID storici e assegniamo nuovi ID solo quando compaiono
    # indirizzi che non erano presenti nel prefisso storico.
    for tx_id, group in inputs.groupby("tx_id", sort=False):
        addresses = sorted(_addresses(group[input_column]))
        known_wallets = {current[address] for address in addresses if address in current}
        if tx_id in blocked:
            for address in addresses:
                current.setdefault(address, new_wallet())
            continue

        if len(known_wallets) == 1:
            # Un solo wallet noto: gli indirizzi nuovi della stessa transazione
            # possono essere estensioni di quel wallet.
            target = next(iter(known_wallets))
            new_addresses = [address for address in addresses if address not in current]
            for address in new_addresses:
                current[address] = target
        elif len(known_wallets) == 0:
            target = new_wallet()
            for address in addresses:
                current[address] = target
        else:
            # Più wallet storici nella stessa transazione: registriamo il
            # conflitto e non li fondiamo automaticamente.
            conflicts.append({
                "tx_id": tx_id,
                "wallet_ids": sorted(known_wallets),
                "addresses": addresses,
            })
            for address in addresses:
                current.setdefault(address, new_wallet())

        for left, right in combinations(addresses, 2):
            if current.get(left) == current.get(right):
                evidence.append({
                    "tx_id": tx_id,
                    "address_a": left,
                    "address_b": right,
                    "reason": "co_spending",
                    "confidence": 1.0,
                })

        for row in candidates_by_tx.get(tx_id, pd.DataFrame()).itertuples(index=False):
            address = str(row.address)
            candidate_wallets = {current[source] for source in addresses if source in current}
            if len(candidate_wallets) == 1:
                target = next(iter(candidate_wallets))
                current.setdefault(address, target)
                for source in addresses:
                    if current.get(source) == target:
                        evidence.append({
                            "tx_id": tx_id,
                            "address_a": min(source, address),
                            "address_b": max(source, address),
                            "reason": "change_address",
                            "confidence": float(row.score),
                        })
            elif len(candidate_wallets) > 1:
                conflicts.append({
                    "tx_id": tx_id,
                    "wallet_ids": sorted(candidate_wallets),
                    "addresses": addresses + [address],
                })
                current.setdefault(address, new_wallet())
            else:
                current.setdefault(address, new_wallet())

    mapping = pd.DataFrame(
        sorted(
            ({"address": address, "wallet_id": wallet_id} for address, wallet_id in current.items()),
            key=lambda row: row["address"],
        ),
        columns=["address", "wallet_id"],
    )
    evidence_frame = pd.DataFrame(
        evidence,
        columns=["tx_id", "address_a", "address_b", "reason", "confidence"],
    ).drop_duplicates()
    conflicts_frame = pd.DataFrame(
        conflicts,
        columns=["tx_id", "wallet_ids", "addresses"],
    )
    return mapping, evidence_frame, conflicts_frame


def complete_mapping_with_observed_addresses(
    mapping: pd.DataFrame,
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
) -> pd.DataFrame:
    """Completa il mapping con tutti gli address osservati nello split.

    Il clustering produce evidenze soprattutto per gli address degli input e
    per i change address. Gli output-only, però, devono comunque comparire
    nel grafo: quando non esiste un'evidenza di fusione li rappresentiamo come
    wallet singleton con un ID deterministico.
    """
    required = {"address", "wallet_id"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"colonne mapping mancanti: {', '.join(sorted(missing))}"
        )

    observed: set[str] = set()
    for frame in (inputs, outputs):
        if "address" in frame.columns:
            observed.update(_addresses(frame["address"]))
        elif "source" in frame.columns:
            observed.update(_addresses(frame["source"]))

    completed = mapping.copy()
    completed["address"] = completed["address"].astype(str)
    completed["wallet_id"] = pd.to_numeric(
        completed["wallet_id"], errors="raise"
    ).astype(int)
    completed = completed.drop_duplicates(subset=["address"], keep="first")
    known = set(completed["address"])
    next_wallet_id = int(completed["wallet_id"].max()) + 1 if not completed.empty else 1

    missing_addresses = sorted(observed - known)
    if missing_addresses:
        additions = pd.DataFrame({
            "address": missing_addresses,
            "wallet_id": range(next_wallet_id, next_wallet_id + len(missing_addresses)),
        })
        # I singleton non hanno evidenze: se il mapping storico contiene una
        # colonna confidence, manteniamo lo stesso schema con valore nullo.
        if "confidence" in completed.columns:
            additions["confidence"] = 0.0
        completed = pd.concat([completed, additions], ignore_index=True)

    return completed.sort_values("address").reset_index(drop=True)
