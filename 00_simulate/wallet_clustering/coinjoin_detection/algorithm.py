"""Prima euristica trasparente per stimare la probabilita di CoinJoin."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Iterable


def classify_transaction(
    n_inputs: int,
    output_amounts: Iterable[Decimal | int | float | str],
) -> dict[str, object]:
    """Restituisce score e segnali CoinJoin per una transazione.

    Lo score e un'euristica: non costituisce un'etichetta definitiva.
    """
    if n_inputs < 0:
        raise ValueError("n_inputs non puo essere negativo")

    # Usiamo Decimal per non perdere il confronto esatto degli importi
    # ripetuti, che e il segnale principale di questa euristica.
    amounts = [Decimal(str(value)) for value in output_amounts]
    n_outputs = len(amounts)
    positive_amounts = [amount for amount in amounts if amount > 0]
    frequencies = Counter(positive_amounts)
    repeated_groups = [count for count in frequencies.values() if count >= 2]
    repeated_outputs = sum(repeated_groups)
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

    # La forma multi-input/multi-output da sola non basta: il punteggio cresce
    # soprattutto quando compaiono output positivi dello stesso valore.
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

    return {
        "score": round(min(score, 1.0), 6),
        "signals": signals,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
    }
