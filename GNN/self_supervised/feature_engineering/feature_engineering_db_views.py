import math

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _prepare_node_features(df_address_features, addresses):
    node_features = pd.DataFrame({"address": addresses})
    node_features = node_features.merge(df_address_features, on="address", how="left")

    feature_columns = [
        "received_output_count",
        "spent_output_count",
        "unspent_output_count",
        "total_received",
        "total_sent",
        "net_flow",
        "estimated_balance",
        "first_seen",
        "last_seen",
        "active_span",
        "activity_count",
        "active_days",
        "region_id",
    ]

    node_features[feature_columns] = node_features[feature_columns].fillna(0.0)

    # Derivate semplici utili per il modello.
    node_features["lifetime_density"] = node_features["activity_count"] / node_features["active_days"].replace(0, 1.0)
    node_features["utxo_ratio"] = node_features["unspent_output_count"] / (
        node_features["received_output_count"].replace(0, 1.0)
    )

    feature_columns.extend(["lifetime_density", "utxo_ratio"])

    features = node_features[feature_columns].copy()

    log_columns = [
        "received_output_count",
        "spent_output_count",
        "unspent_output_count",
        "total_received",
        "total_sent",
        "estimated_balance",
        "active_span",
        "activity_count",
        "active_days",
        "lifetime_density",
    ]

    for column in log_columns:
        features[column] = np.log1p(features[column].clip(lower=0.0))

    features["net_flow"] = np.sign(features["net_flow"]) * np.log1p(np.abs(features["net_flow"]))
    features["region_id"] = features["region_id"] / 12.0
    features["utxo_ratio"] = features["utxo_ratio"].clip(lower=0.0, upper=1.0)

    means = features.mean(axis=0)
    stds = features.std(axis=0, ddof=0).replace(0, 1.0)
    normalized = (features - means) / stds

    return normalized, feature_columns


def build_graph_features_from_views(df_flows, df_address_features):
    addresses = pd.concat([df_flows.from_address, df_flows.to_address]).dropna().unique()
    addr_to_idx = {addr: i for i, addr in enumerate(addresses)}

    edge_df = df_flows[df_flows.from_address.notna() & df_flows.to_address.notna()].copy()
    src = edge_df.from_address.map(addr_to_idx).values
    dst = edge_df.to_address.map(addr_to_idx).values

    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    edge_time = torch.tensor(edge_df.time.values, dtype=torch.long)

    feature_matrix, feature_names = _prepare_node_features(df_address_features, addresses)

    data = Data(
        x=torch.tensor(feature_matrix.values, dtype=torch.float),
        edge_index=edge_index,
    )
    data.edge_time = edge_time
    data.feature_names = feature_names

    return data


def build_graph_features_from_views_for_anomaly(df_flows, df_address_features):
    addresses = pd.concat([df_flows.from_address, df_flows.to_address]).dropna().unique()
    idx_to_address = {i: addr for i, addr in enumerate(addresses)}

    data = build_graph_features_from_views(df_flows, df_address_features)
    data.idx_to_address = idx_to_address

    return data
