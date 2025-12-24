import json
from datetime import datetime

def save_anomaly_report(
    *,
    addresses,
    norms,
    indices_sospetti,
    df_edges,
    classify_fn,
    threshold,
    num_nodes,
    output_path=".",
    prefix="bitcoin_anomaly_report"
):
    """
    Salva un report JSON con i dettagli delle anomalie.

    Parameters
    ----------
    addresses : list[str]
        Lista degli indirizzi indicizzati per nodo
    norms : torch.Tensor
        Score di anomalia per ogni nodo
    indices_sospetti : list[int]
        Indici dei nodi anomali
    df_edges : pd.DataFrame
        DataFrame degli archi (flows)
    classify_fn : callable
        Funzione classify_node_with_scores(addr, df_edges)
    threshold : float
        Soglia di anomalia usata
    num_nodes : int
        Numero totale di nodi
    output_path : str
        Directory di output
    prefix : str
        Prefisso nome file
    """

    output = {
        "model": {
            "type": "VGAE",
            "method": "MAD",
            "threshold": float(threshold)
        },
        "summary": {
            "num_nodes": int(num_nodes),
            "num_edges": int(len(df_edges)),
            "num_anomalies": len(indices_sospetti)
        },
        "anomalies": []
    }

    for i in indices_sospetti:
        addr = addresses[i]
        score = norms[i].item()

        reasons = classify_fn(addr, df_edges)

        node_edges = df_edges[
            (df_edges["from_address"] == addr) |
            (df_edges["to_address"] == addr)
        ]

        out_edges = node_edges[node_edges["from_address"] == addr]
        in_edges = node_edges[node_edges["to_address"] == addr]

        anomaly_entry = {
            "address": addr,
            "anomaly_score": round(score, 6),
            "reasons": [
                {"label": label, "score": round(score_, 4)}
                for label, score_ in reasons
            ],
            "stats": {
                "out_degree": int(len(out_edges)),
                "out_sum": float(out_edges["flow_amount"].sum()),
                "in_degree": int(len(in_edges)),
                "in_sum": float(in_edges["flow_amount"].sum())
            },
            "edges": node_edges[
                ["txid", "from_address", "to_address", "flow_amount", "time"]
            ].to_dict(orient="records")
        }

        output["anomalies"].append(anomaly_entry)

    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    full_path = f"{output_path}/{filename}"

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return full_path