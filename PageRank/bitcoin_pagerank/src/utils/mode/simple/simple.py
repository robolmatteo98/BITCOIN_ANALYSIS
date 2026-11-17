import math

def value_weight(edge):
    transactions = edge.get("transactions") or []
    if not transactions:
        return 0.0

    raw_value = sum(float(transaction.get("value") or 0.0) for transaction in transactions)
    value = max(float(raw_value or 0.0), 0.0)
    return math.log1p(value)

def get_edge_timestamp(edge):
    transactions = edge.get("transactions") or []
    if not transactions:
        return None

    timestamps = [
        float(transaction["timestamp"])
        for transaction in transactions
        if transaction.get("timestamp") is not None
    ]
    if not timestamps:
        return None
    return sum(timestamps) / len(timestamps)

def normalize_timestamp(timestamp, min_ts, max_ts):
    if timestamp is None or min_ts is None or max_ts is None:
        return None
    if max_ts <= min_ts:
        return 1.0
    value = float(timestamp) / float(max_ts)
    return max(0.0, min(1.0, value))

def time_weight(edge, get_timestamp, time_bounds):
    timestamp = get_timestamp(edge)
    if timestamp is None or time_bounds is None:
        return 1.0
    min_ts, max_ts = time_bounds
    recency = normalize_timestamp(timestamp, min_ts, max_ts)
    return float(recency) if recency is not None else 1.0

def edge_weight(edge, mode, time_bounds=None):
    if mode == "basic":
        return 1.0
    if mode == "combined":
        return value_weight(edge) * time_weight(edge, get_edge_timestamp, time_bounds)
    raise ValueError(f"Unsupported mode: {mode}")