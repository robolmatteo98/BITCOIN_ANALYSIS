"""Prima euristica per identificare possibili change address."""

from __future__ import annotations

import pandas as pd


REQUIRED_INPUT_COLUMNS = {"tx_id", "address", "amount"}
REQUIRED_OUTPUT_COLUMNS = {"tx_id", "output_index", "address", "amount"}


def find_change_candidates(
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
    known_addresses: set[str] | None = None,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """Restituisce gli output che sembrano indirizzi di resto.

    Questa prima versione non usa script Bitcoin. Il punteggio combina:
    indirizzo nuovo, indirizzo diverso dagli input, posizione finale e
    importo inferiore al totale degli input.
    """

    # Controllo dei campi
    missing_inputs = REQUIRED_INPUT_COLUMNS - set(inputs.columns)
    missing_outputs = REQUIRED_OUTPUT_COLUMNS - set(outputs.columns)
    if missing_inputs:
        raise ValueError(f"colonne input mancanti: {', '.join(sorted(missing_inputs))}")
    if missing_outputs:
        raise ValueError(
            f"colonne output mancanti: {', '.join(sorted(missing_outputs))}"
        )
    if not 0 <= threshold <= 1:
        raise ValueError("threshold deve essere compresa tra 0 e 1")

    # Faccio una copia di input e output
    # 
    input_rows = inputs.copy()
    output_rows = outputs.copy()
    input_rows["amount"] = pd.to_numeric(input_rows["amount"], errors="raise")
    output_rows["amount"] = pd.to_numeric(output_rows["amount"], errors="raise")
    output_rows["output_index"] = pd.to_numeric(
        output_rows["output_index"], errors="raise"
    ).astype(int)

    # con known_addresses inizialmente vuoto
    seen_addresses = set(known_addresses or set())
    results = []
    transaction_order = (
        input_rows[["tx_id", "timestamp"]]
        .drop_duplicates("tx_id")
        .sort_values("timestamp")["tx_id"].tolist()
        if "timestamp" in input_rows.columns
        else input_rows["tx_id"].drop_duplicates().tolist()
    )

    print(transaction_order)

    for tx_id in transaction_order:

        # prende tutti gli input e output della transazione tx_id
        tx_inputs = input_rows[input_rows["tx_id"] == tx_id]
        tx_outputs = output_rows[output_rows["tx_id"] == tx_id].copy()
        if tx_inputs.empty or tx_outputs.empty:
            continue

        # viene creato un insieme di indirizzi distinti
        input_addresses = {
            str(address)
            for address in tx_inputs["address"]
            if pd.notna(address)
        }
        
        tx_input_total = float(tx_inputs["amount"].sum())

        # se indirizzo dell'output non è presente in input = true, altrimenti false
        non_input_mask = tx_outputs["address"].astype(str).isin(input_addresses) == False

        # conta quanti output hanno un indirizzo non presente tra gli input
        non_input_count = int(non_input_mask.sum())

        # trova l'indice numerico più alto (dovrebbe essere l'ultimo della transazione)
        last_output_index = int(tx_outputs["output_index"].max())

        # copia degli indirizzi visti e aggiornati con nuovi indirizzi di questa transazione
        seen_before_outputs = set(seen_addresses)
        seen_addresses.update(input_addresses)

        for row in tx_outputs.itertuples(index=False):
            address = str(row.address)

            # verifica se l'indirizzo non era già stato visto prima degli output correnti
            # verifica se l'indirizzo non era un indirizzo di input
            is_new = address not in seen_before_outputs
            is_not_input = address not in input_addresses

            # controlla se l'output corrente è l'ultimo della transazione
            is_last_output = int(row.output_index) == last_output_index

            # controlla se l'importo dell'output corrente è minore del totale degli input
            amount_below_inputs = float(row.amount) < tx_input_total

            # 1. l’indirizzo dell’output non compare tra gli indirizzi degli input (is_not_input)
            # 2. esiste un solo output con questa caratteristica (non_input_count == 1)
            only_non_input_output = (
                is_not_input and non_input_count == 1
            )

            score = (
                0.35 * is_new
                + 0.15 * is_not_input
                + 0.20 * is_last_output
                + 0.15 * only_non_input_output
                + 0.15 * amount_below_inputs
            )
            reasons = []
            if is_new:
                reasons.append("indirizzo nuovo")
            if is_not_input:
                reasons.append("non compare tra gli input")
            if is_last_output:
                reasons.append("ultimo output")
            if only_non_input_output:
                reasons.append("unico output non-input")
            if amount_below_inputs:
                reasons.append("importo minore del totale input")

            results.append({
                "tx_id": tx_id,
                "output_index": int(row.output_index),
                "address": address,
                "amount": float(row.amount),
                "score": round(score, 6),
                "is_change_candidate": score >= threshold,
                "reasons": reasons,
            })

        seen_addresses.update(
            address for address in tx_outputs["address"].astype(str)
        )

    return pd.DataFrame(
        results,
        columns=[
            "tx_id", "output_index", "address", "amount", "score",
            "is_change_candidate", "reasons",
        ],
    ).sort_values(
        ["is_change_candidate", "score", "tx_id", "output_index"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
