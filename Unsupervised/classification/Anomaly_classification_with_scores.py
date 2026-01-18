import pandas as pd
import numpy as np

def classify_node_with_scores(address: str, df_edges: pd.DataFrame, top_k=3):
    """
    Restituisce una classifica delle possibili categorie
    con score normalizzati (0–1).
    """

    node_edges = df_edges[
        (df_edges["from_address"] == address) |
        (df_edges["to_address"] == address)
    ]

    if len(node_edges) == 0:
        return [("Inactive / no observed behavior", 1.0)]

    out_edges = node_edges[node_edges["from_address"] == address]
    in_edges  = node_edges[node_edges["to_address"] == address]

    out_deg = len(out_edges)
    in_deg  = len(in_edges)

    out_sum = out_edges["flow_amount"].sum()
    in_sum  = in_edges["flow_amount"].sum()

    times_out = np.sort(out_edges["time"].values)
    times_in  = np.sort(in_edges["time"].values)

    scores = {
        "Peeling chain behavior": 0.0,
        "Fund distribution hub": 0.0,
        "Fund aggregation node": 0.0,
        "Self-churn / cyclic": 0.0,
        "Mixer-like behavior": 0.0,
        "Exchange / service wallet": 0.0,
    }

    # --------------------------------------------------
    # Aggregation / collection
    # --------------------------------------------------
    if in_deg >= 5 and out_deg <= 2:
        scores["Fund aggregation node"] += min(1.0, in_deg / 20)

    # --------------------------------------------------
    # Exchange-like
    # --------------------------------------------------
    if in_deg >= 10:
        balance_ratio = abs(in_sum - out_sum) / max(in_sum, 1e-9)
        scores["Exchange / service wallet"] += max(0, 1 - balance_ratio)

    # --------------------------------------------------
    # Peeling chain
    # --------------------------------------------------
    if (
        out_deg >= 2 and
        in_deg <= 1 and
        out_sum <= in_sum * 1.05 and
        len(times_out) >= 2 and
        np.max(np.diff(times_out)) < 10_000
    ):
        scores["Peeling chain behavior"] += 0.8

    # --------------------------------------------------
    # Distribution hub
    # --------------------------------------------------
    if out_deg >= 5 and in_deg <= 2:
        scores["Fund distribution hub"] += min(1.0, out_deg / 20)

    # --------------------------------------------------
    # Mixer-like
    # --------------------------------------------------
    if (
        in_deg >= 3 and out_deg >= 3 and
        abs(in_sum - out_sum) / max(in_sum, 1e-9) < 0.15
    ):
        scores["Mixer-like behavior"] += 0.6

    # --------------------------------------------------
    # Self churn
    # --------------------------------------------------
    mutuals = set(out_edges["to_address"]).intersection(
        set(in_edges["from_address"])
    )
    if mutuals:
        scores["Self-churn / cyclic"] += 0.5

    # --------------------------------------------------
    # Normalizzazione
    # --------------------------------------------------
    max_score = max(scores.values())
    if max_score > 0:
        for k in scores:
            scores[k] /= max_score

    # Ordina e restituisce top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = [(k, round(v, 3)) for k, v in ranked if v > 0]

    if not ranked:
        return [("Irregular / anomalous behavior", 1.0)]

    return ranked[:top_k]