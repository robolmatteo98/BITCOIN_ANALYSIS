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

    # Controllo dei campi. Lo script type è opzionale negli esempi, ma viene
    # usato quando la funzione riceve i dati completi dal database.
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

    # In ogni momento conosciamo solo gli indirizzi osservati prima della
    # transazione corrente: così la freschezza non guarda nel futuro.
    seen_addresses = set(known_addresses or set())
    results = []
    transaction_order = (
        input_rows[["tx_id", "timestamp"]]
        .drop_duplicates("tx_id")
        .sort_values("timestamp")["tx_id"].tolist()
        if "timestamp" in input_rows.columns
        else input_rows["tx_id"].drop_duplicates().tolist()
    )

    if "script_type" not in input_rows.columns:
        input_rows["script_type"] = "UNKNOWN"
    if "script_type" not in output_rows.columns:
        output_rows["script_type"] = "UNKNOWN"

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
        
        tx_input_amounts = pd.to_numeric(tx_inputs["amount"], errors="raise")
        tx_input_total = float(tx_input_amounts.sum())
        positive_input_amounts = [
            float(amount) for amount in tx_input_amounts if amount > 0
        ]
        input_script_types = [str(value) for value in tx_inputs["script_type"]]
        known_input_script_types = [
            value for value in input_script_types if value != "UNKNOWN"
        ]
        uniform_input_script_type = (
            len(known_input_script_types) == len(input_script_types)
            and len(set(known_input_script_types)) == 1
        )

        # se indirizzo dell'output non è presente in input = true, altrimenti false
        non_input_mask = tx_outputs["address"].astype(str).isin(input_addresses) == False

        # conta quanti output hanno un indirizzo non presente tra gli input
        non_input_count = int(non_input_mask.sum())

        # trova l'indice numerico più alto (dovrebbe essere l'ultimo della transazione)
        last_output_index = int(tx_outputs["output_index"].max())

        # copia degli indirizzi visti e aggiornati con nuovi indirizzi di questa transazione
        seen_before_outputs = set(seen_addresses)
        seen_addresses.update(input_addresses)

        candidate_rows = []
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
            single_output_full_value = (
                len(tx_outputs) == 1
                and is_not_input
                and not amount_below_inputs
            )
            optimal_change = (
                len(positive_input_amounts) >= 2
                and float(row.amount) < min(positive_input_amounts)
            )
            single_input_change = (
                len(positive_input_amounts) == 1
                and amount_below_inputs
            )
            optimal_component = (
                0.30 if optimal_change
                else 0.12 if single_input_change
                else 0.0
            )
            output_script_type = str(row.script_type)
            script_type_match = (
                uniform_input_script_type
                and output_script_type != "UNKNOWN"
                and output_script_type == known_input_script_types[0]
            )

            # Un output nuovo e diverso dagli input è solo un candidato: anche
            # un normale destinatario può avere esattamente queste proprietà.
            only_non_input_output = (
                is_not_input and non_input_count == 1
            )

            # Optimal change e freschezza sono i segnali principali. Posizione
            # e unicita dell'output sono invece segnali deboli: descrivono
            # anche molti pagamenti normali.
            score = (
                optimal_component
                + 0.20 * is_new
                + 0.20 * script_type_match
                + 0.10 * amount_below_inputs
                + 0.08 * only_non_input_output
                + 0.04 * is_last_output
                + 0.03 * is_not_input
                + 0.05 * (len(tx_outputs) >= 2)
            )
            # Un solo output che assorbe tutto il valore puo essere una
            # consolidazione o un trasferimento interno verso un nuovo
            # indirizzo. E un caso possibile, ma non un forte segnale di
            # change: si conserva con un contributo basso e non supera la
            # soglia standard.
            if single_output_full_value:
                score = 0.10
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
            if optimal_change:
                reasons.append("importo minore di ogni input")
            if script_type_match:
                reasons.append("stesso tipo di script degli input")
            elif single_input_change:
                reasons.append("resto plausibile con un solo input")
            if single_output_full_value:
                reasons.append("unico output con valore totale degli input")

            candidate_rows.append({
                "tx_id": tx_id,
                "output_index": int(row.output_index),
                "address": address,
                "amount": float(row.amount),
                "script_type": output_script_type,
                "score": round(score, 6),
                "is_change_candidate": False,
                "reasons": reasons,
            })

        # Se piu output sono ugualmente plausibili, non scegliamo nessuno:
        # un falso positivo puo propagarsi nel clustering in modo transitivo.
        if candidate_rows and len(tx_outputs) >= 2:
            best_score = max(row["score"] for row in candidate_rows)
            best_rows = [
                row for row in candidate_rows
                if abs(row["score"] - best_score) < 1e-9
            ]
            if len(best_rows) == 1 and best_score >= threshold:
                best_rows[0]["is_change_candidate"] = True
        results.extend(candidate_rows)

        seen_addresses.update(
            address for address in tx_outputs["address"].astype(str)
        )

    return pd.DataFrame(
        results,
        columns=[
            "tx_id", "output_index", "address", "amount", "script_type", "score",
            "is_change_candidate", "reasons",
        ],
    ).sort_values(
        ["is_change_candidate", "score", "tx_id", "output_index"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
