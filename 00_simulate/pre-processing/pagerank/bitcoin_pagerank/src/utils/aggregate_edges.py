def edge_to_aggregation_record(edge):
    value_raw = edge.get("value_sat")
    timestamp_raw = edge.get("timestamp")

    value = float(value_raw or 0.0)
    timestamp = int(timestamp_raw) if timestamp_raw is not None else None

    return (
        (edge["src"], edge["dst"]),
        {
            "tx_count": 1,
            "transactions": [
                {
                    "value": value, 
                    "timestamp": timestamp,
                }
            ],
        },
    )

# ATTENZIONE: la concatenazione di due liste occupa tanto spazio in memoria, quando inziano ad essere piuttosto grandi
# possibile ottimizzazione
def merge_aggregation_records(left, right):
    return {
        "tx_count": left["tx_count"] + right["tx_count"],
        "transactions": left["transactions"] + right["transactions"],
    }

def aggregation_record_to_edge(item):
    (src, dst), values = item

    return {
        "src": src,
        "dst": dst,
        "tx_count": values["tx_count"],
        "transactions": values["transactions"],
    }

def aggregate_edges_by_pair(edges_rdd):
    return (
        edges_rdd
        .map(edge_to_aggregation_record)
        .reduceByKey(merge_aggregation_records)
        .map(aggregation_record_to_edge)
    )