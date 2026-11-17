import csv

def extract_entity(script_pub_key):
    if not script_pub_key:
        return None
    return script_pub_key.get("entity") or script_pub_key.get("address")

def block_to_edge_rows(block):
    timestamp = block.get("time")
    edges = []

    for tx in block.get("tx", []):
        inputs = []
        for vin in tx.get("vin", []):
            src = vin.get("prevout_entity") or vin.get("prevout_address")
            if src:
                inputs.append(src)

        outputs = []
        for vout in tx.get("vout", []):
            dst = extract_entity(vout.get("scriptPubKey"))
            if dst:
                outputs.append((dst, int(vout.get("value_sat", 0) or 0)))

        if not inputs or not outputs:
            continue

        for src in set(inputs):
            for dst, value_sat in outputs:
                if src == dst:
                    continue
                edges.append({
                    "src": src,
                    "dst": dst,
                    "value_sat": value_sat,
                    "timestamp": int(timestamp) if timestamp is not None else None,
                })

    return edges

def parse_edge_csv_line(line):
    parts = next(csv.reader([line]))
    if not parts or parts[0] == "src":
        return None

    return {
        "src": parts[0],
        "dst": parts[1],
        "value_sat": float(parts[2]) if parts[2] else 0.0,
        "timestamp": int(parts[3]) if len(parts) > 3 and parts[3] else None,
    }