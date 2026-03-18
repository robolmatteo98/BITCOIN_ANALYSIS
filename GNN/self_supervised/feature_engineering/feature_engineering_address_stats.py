import datetime
import math

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


ACTIVE_HOURS = set(range(10, 23))


def _estimate_utc_offset(hours):
    if not hours:
        return 0

    best_offset = 0
    best_score = -1

    for offset in range(-12, 13):
        shifted_hours = [int((hour + offset) % 24) for hour in hours]
        score = sum(1 for hour in shifted_hours if hour in ACTIVE_HOURS)

        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset


def _hour_entropy(hours):
    if not hours:
        return 0.0

    counts = np.bincount(hours, minlength=24).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = math.log2(24)

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _build_temporal_features(df):
    address_times = {}

    for row in df[["from_address", "to_address", "time"]].itertuples(index=False):
        from_address, to_address, timestamp = row

        if pd.notna(from_address):
            address_times.setdefault(from_address, []).append(int(timestamp))

        if pd.notna(to_address):
            address_times.setdefault(to_address, []).append(int(timestamp))

    rows = []

    for address, timestamps in address_times.items():
        timestamps = sorted(timestamps)
        hours = [datetime.datetime.utcfromtimestamp(ts).hour for ts in timestamps]
        utc_offset = _estimate_utc_offset(hours)
        shifted_hours = [(hour + utc_offset) % 24 for hour in hours]
        daylight_ratio = sum(1 for hour in shifted_hours if hour in ACTIVE_HOURS) / len(shifted_hours)

        rows.append(
            {
                "address": address,
                "first_seen": timestamps[0],
                "last_seen": timestamps[-1],
                "active_span": timestamps[-1] - timestamps[0],
                "active_days": len({ts // 86400 for ts in timestamps}),
                "incident_tx_count": len(timestamps),
                "hour_entropy": _hour_entropy(hours),
                "estimated_utc_offset": utc_offset,
                "daylight_activity_ratio": daylight_ratio,
            }
        )

    return pd.DataFrame(rows)


def _build_node_feature_frame(df, addresses):
    outgoing = (
        df.groupby("from_address")
        .agg(
            out_degree=("to_address", "nunique"),
            tx_count_out=("to_address", "size"),
            out_amount_sum=("flow_amount", "sum"),
            out_amount_mean=("flow_amount", "mean"),
            out_amount_max=("flow_amount", "max"),
            out_amount_std=("flow_amount", "std"),
        )
        .reset_index()
        .rename(columns={"from_address": "address"})
    )

    incoming = (
        df.groupby("to_address")
        .agg(
            in_degree=("from_address", "nunique"),
            tx_count_in=("from_address", "size"),
            in_amount_sum=("flow_amount", "sum"),
            in_amount_mean=("flow_amount", "mean"),
            in_amount_max=("flow_amount", "max"),
            in_amount_std=("flow_amount", "std"),
        )
        .reset_index()
        .rename(columns={"to_address": "address"})
    )

    temporal = _build_temporal_features(df)

    node_features = pd.DataFrame({"address": addresses})
    node_features = node_features.merge(outgoing, on="address", how="left")
    node_features = node_features.merge(incoming, on="address", how="left")
    node_features = node_features.merge(temporal, on="address", how="left")

    fill_zero_columns = [
        "out_degree",
        "tx_count_out",
        "out_amount_sum",
        "out_amount_mean",
        "out_amount_max",
        "out_amount_std",
        "in_degree",
        "tx_count_in",
        "in_amount_sum",
        "in_amount_mean",
        "in_amount_max",
        "in_amount_std",
        "first_seen",
        "last_seen",
        "active_span",
        "active_days",
        "incident_tx_count",
        "hour_entropy",
        "estimated_utc_offset",
        "daylight_activity_ratio",
    ]

    node_features[fill_zero_columns] = node_features[fill_zero_columns].fillna(0.0)
    node_features["net_flow"] = node_features["in_amount_sum"] - node_features["out_amount_sum"]

    return node_features


def _prepare_feature_matrix(node_features):
    feature_columns = [
        "out_degree",
        "in_degree",
        "tx_count_out",
        "tx_count_in",
        "incident_tx_count",
        "out_amount_sum",
        "in_amount_sum",
        "net_flow",
        "out_amount_mean",
        "in_amount_mean",
        "out_amount_max",
        "in_amount_max",
        "out_amount_std",
        "in_amount_std",
        "active_span",
        "active_days",
        "hour_entropy",
        "estimated_utc_offset",
        "daylight_activity_ratio",
    ]

    features = node_features[feature_columns].copy()

    log_columns = [
        "out_degree",
        "in_degree",
        "tx_count_out",
        "tx_count_in",
        "incident_tx_count",
        "out_amount_sum",
        "in_amount_sum",
        "out_amount_mean",
        "in_amount_mean",
        "out_amount_max",
        "in_amount_max",
        "out_amount_std",
        "in_amount_std",
        "active_span",
        "active_days",
    ]

    for column in log_columns:
        features[column] = np.log1p(features[column].clip(lower=0.0))

    # Mantiene il segno del saldo netto ma ne comprime la scala.
    features["net_flow"] = np.sign(features["net_flow"]) * np.log1p(np.abs(features["net_flow"]))

    means = features.mean(axis=0)
    stds = features.std(axis=0, ddof=0).replace(0, 1.0)
    normalized = (features - means) / stds

    return normalized, feature_columns


def build_graph_features(df):
    addresses = pd.concat([df.from_address, df.to_address]).dropna().unique()
    addr_to_idx = {addr: i for i, addr in enumerate(addresses)}

    edge_df = df[df.from_address.notna() & df.to_address.notna()].copy()
    src = edge_df.from_address.map(addr_to_idx).values
    dst = edge_df.to_address.map(addr_to_idx).values

    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    edge_time = torch.tensor(edge_df.time.values, dtype=torch.long)

    node_features = _build_node_feature_frame(edge_df, addresses)
    feature_matrix, feature_names = _prepare_feature_matrix(node_features)

    data = Data(
        x=torch.tensor(feature_matrix.values, dtype=torch.float),
        edge_index=edge_index,
    )
    data.edge_time = edge_time
    data.feature_names = feature_names

    return data


def build_graph_features_for_anomaly(df):
    addresses = pd.concat([df.from_address, df.to_address]).dropna().unique()
    idx_to_address = {i: addr for i, addr in enumerate(addresses)}

    data = build_graph_features(df)
    data.idx_to_address = idx_to_address

    return data
