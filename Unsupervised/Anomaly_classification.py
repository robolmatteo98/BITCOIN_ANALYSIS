def classify_suspicious_node(addr, df):
    edges = df[(df["from_address"] == addr) | (df["to_address"] == addr)]

    out_edges = edges[edges["from_address"] == addr]
    in_edges  = edges[edges["to_address"] == addr]

    out_deg = len(out_edges)
    in_deg = len(in_edges)

    # ==========================
    # 1) HUB / SPRAY PATTERN
    # ==========================
    if out_deg >= 5 and in_deg <= 2:
        return "Hub/Spray (many outputs) — possible mixer or fund splitting"

    # ==========================
    # 2) PEELING CHAIN
    # ==========================
    if out_deg == 1 and in_deg == 1:
        amounts = out_edges["flow_amount"].values
        if len(amounts) == 1:
            return "Peeling chain step (1-in 1-out transaction with structured amount)"

    # ==========================
    # 3) SELF SHUFFLE
    # ==========================
    # check if both outgoing and incoming with same address
    for _, row in out_edges.iterrows():
        if ((df["from_address"] == row["to_address"]) & 
            (df["to_address"] == addr)).any():
            return "Self-shuffle / round-trip — mixer-like pattern"

    # ==========================
    # 4) TOPOLOGICAL OUTLIER
    # ==========================
    if out_deg + in_deg <= 1:
        return "Topological outlier — isolated or rare-flow node"

    # default
    return "Generic anomaly — unusual embedding"