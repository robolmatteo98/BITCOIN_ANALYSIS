def normal_wallet_template(prefix, t0):
    return [
        {
            "from_address": f"{prefix}_1",
            "to_address": f"{prefix}_2",
            "flow_amount": 50.0,
            "time": t0,
            "n_inputs": 1,
            "n_outputs": 1,
            "total_amount": 50.0
        },
        {
            "from_address": f"{prefix}_1",
            "to_address": f"{prefix}_3",
            "flow_amount": 25.0,
            "time": t0 + 1,
            "n_inputs": 1,
            "n_outputs": 2,
            "total_amount": 50.0
        },
        {
            "from_address": f"{prefix}_1",
            "to_address": f"{prefix}_4",
            "flow_amount": 25.0,
            "time": t0 + 1,
            "n_inputs": 1,
            "n_outputs": 2,
            "total_amount": 50.0
        },
        {
            "from_address": f"{prefix}_2",
            "to_address": f"{prefix}_5",
            "flow_amount": 0.1,
            "time": t0 + 2,
            "n_inputs": 2,
            "n_outputs": 1,
            "total_amount": 50.0
        },
        {
            "from_address": f"{prefix}_1",
            "to_address": f"{prefix}_5",
            "flow_amount": 49.9,
            "time": t0 + 2,
            "n_inputs": 2,
            "n_outputs": 1,
            "total_amount": 50.0
        },
        {
            "from_address": f"{prefix}_5",
            "to_address": f"{prefix}_6",
            "flow_amount": 32.003,
            "time": t0 + 3,
            "n_inputs": 1,
            "n_outputs": 1,
            "total_amount": 32.003
        },
        {
            "from_address": f"{prefix}_3",
            "to_address": f"{prefix}_5",
            "flow_amount": 4.2,
            "time": t0 + 4,
            "n_inputs": 1,
            "n_outputs": 1,
            "total_amount": 4.3
        },
        {
            "from_address": f"{prefix}_5",
            "to_address": f"{prefix}_2",
            "flow_amount": 10.05,
            "time": t0 + 6,
            "n_inputs": 1,
            "n_outputs": 1,
            "total_amount": 10.05
        },
        {
            "from_address": f"{prefix}_2",
            "to_address": f"{prefix}_7",
            "flow_amount": 40.3,
            "time": t0 + 7,
            "n_inputs": 1,
            "n_outputs": 1,
            "total_amount": 40.3
        },
        {
            "from_address": f"{prefix}_6",
            "to_address": f"{prefix}_4",
            "flow_amount": 32,
            "time": t0 + 8,
            "n_inputs": 1,
            "n_outputs": 2,
            "total_amount": 32.003
        },
        {
            "from_address": f"{prefix}_6",
            "to_address": f"{prefix}_6",
            "flow_amount": 0.003,
            "time": t0 + 8,
            "n_inputs": 1,
            "n_outputs": 2,
            "total_amount": 0.003
        },
    ]


def fake_load_data():
    rows = []

    # 20 wallet normali → ~220 nodi
    for i in range(20):
        rows.extend(normal_wallet_template(
            prefix=f"a{i}",
            t0=i * 100
        ))

    # 🔴 super hub globale (anomalia)
    for i in range(20):
        rows.append({
            "from_address": "a_hub",
            "to_address": f"a{i}_1",
            "flow_amount": 1000,
            "time": 9999,
            "n_inputs": 1,
            "n_outputs": 50,
            "total_amount": 1000
        })

    df = pd.DataFrame(rows)
    addresses = sorted(set(df.from_address) | set(df.to_address))
    return df, addresses