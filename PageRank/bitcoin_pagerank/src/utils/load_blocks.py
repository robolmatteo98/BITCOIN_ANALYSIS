from bitcoin_common.shared_bitcoin_graph import parse_edge_csv_line

def load_edges_csv_rdd(sc, input_glob):
    # textFile legge tutte le partizioni del CSV distribuito prodotto da Spark.
    lines_rdd = sc.textFile(input_glob)

    # Filtriamo l'header e trasformiamo ogni riga in un dizionario edge.
    edges_rdd = lines_rdd.map(parse_edge_csv_line).filter(lambda row: row is not None)
    return edges_rdd

def compute_time_bounds(edges_rdd):
    timestamps_rdd = (
        edges_rdd
        .map(lambda edge: float(edge["timestamp"]) if edge.get("timestamp") is not None else None)
        .filter(lambda timestamp: timestamp is not None)
        .cache()
    )
    if timestamps_rdd.isEmpty():
        return None, None
    return timestamps_rdd.min(), timestamps_rdd.max()