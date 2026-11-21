"""Prima euristica trasparente per stimare la probabilità di CoinJoin.

La funzione non pretende di identificare con certezza una CoinJoin. Usa segnali
disponibili nello schema attuale: numero di input/output e quantità ripetute
negli output. Il risultato è quindi un punteggio da usare come feature o filtro,
non un'etichetta definitiva.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Iterable


def classify_transaction(
    n_inputs: int,
    output_amounts: Iterable[Decimal | int | float | str],
) -> dict[str, object]:
    """Restituisce lo score e i segnali calcolati per una transazione.

    Gli importi sono convertiti in ``Decimal`` e confrontati esattamente. Gli
    output con importo zero non vengono usati per il segnale degli importi
    uguali, perché possono rappresentare OP_RETURN o output non monetari.
    """

    if n_inputs < 0:
        raise ValueError("n_inputs non può essere negativo")

    # converto gli amounts in decimali
    # conto quanti sono
    # seleziono solo i positivi
    # conta quante volte compare un elemento nella lista: chiave (elemento) - valore (counter)
    amounts = [Decimal(str(value)) for value in output_amounts]
    n_outputs = len(amounts)
    positive_amounts = [amount for amount in amounts if amount > 0]
    frequencies = Counter(positive_amounts)

    # filtra quanti counter sono maggiori di 2 e li somma
    repeated_groups = [count for count in frequencies.values() if count >= 2]

    # Indica quanti output appartengono a gruppi di importi ripetuti
    repeated_outputs = sum(repeated_groups)

    # Indica quanti output appartengono a gruppi di importi ripetuti
    largest_equal_group = max(repeated_groups, default=0)

    signals = {
        "has_multiple_inputs_outputs": n_inputs >= 2 and n_outputs >= 2,
        "has_participant_shape": n_inputs >= 3 and n_outputs >= 3,
        "has_repeated_output_amount": bool(repeated_groups),
        "repeated_output_share": (
            repeated_outputs / len(positive_amounts) if positive_amounts else 0.0
        ),
        "repeated_amount_groups": len(repeated_groups),
        "largest_equal_group": largest_equal_group,
    }

    # Pesi espliciti e facilmente modificabili. La ripetizione degli importi
    # è il segnale principale; la forma della transazione da sola non basta.
    score = 0.0
    if signals["has_multiple_inputs_outputs"]:
        score += 0.15
    if signals["has_participant_shape"]:
        score += 0.20
    if signals["has_repeated_output_amount"]:
        score += 0.35
    if signals["repeated_output_share"] >= 0.5:
        score += 0.20
    if signals["repeated_amount_groups"] >= 2:
        score += 0.10

    score = round(min(score, 1.0), 6)
    return {
        "score": score,
        "signals": signals,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
    }
