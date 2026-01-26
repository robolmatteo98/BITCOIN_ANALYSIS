import pandas as pd
import numpy as np

def fake_load_data():
    """
    Simula flows_view
    """

    rows = [
      {
        "from_address": "a_1",
        "to_address": "a_2",
        "flow_amount": 50.0,
        "time": 0,
        "n_inputs": 1,
        "n_outputs": 1,
        "total_amount": 50.0
      },
      {
        "from_address": "a_1",
        "to_address": "a_3",
        "flow_amount": 25.0,
        "time": 1,
        "n_inputs": 1,
        "n_outputs": 2,
        "total_amount": 50.0
      },
      {
        "from_address": "a_1",
        "to_address": "a_4",
        "flow_amount": 25.0,
        "time": 1,
        "n_inputs": 1,
        "n_outputs": 2,
        "total_amount": 50.0
      },
      {
        "from_address": "a_2",
        "to_address": "a_5",
        "flow_amount": 0.1,
        "time": 2,
        "n_inputs": 2,
        "n_outputs": 1,
        "total_amount": 50.0
      },
      {
        "from_address": "a_1",
        "to_address": "a_5",
        "flow_amount": 49.9,
        "time": 2,
        "n_inputs": 2,
        "n_outputs": 1,
        "total_amount": 50.0
      },
      {
        "from_address": "a_5",
        "to_address": "a_6",
        "flow_amount": 32.003,
        "time": 3,
        "n_inputs": 1,
        "n_outputs": 1,
        "total_amount": 32.003
      },
      {
        "from_address": "a_3",
        "to_address": "a_5",
        "flow_amount": 4.2,
        "time": 4,
        "n_inputs": 1,
        "n_outputs": 1,
        "total_amount": 4.3
      },
      {
        "from_address": "a_5",
        "to_address": "a_2",
        "flow_amount": 10.05,
        "time": 6,
        "n_inputs": 1,
        "n_outputs": 1,
        "total_amount": 10.05
      },
      {
        "from_address": "a_2",
        "to_address": "a_7",
        "flow_amount": 40.3,
        "time": 7,
        "n_inputs": 1,
        "n_outputs": 1,
        "total_amount": 40.3
      },
      {
        "from_address": "a_6",
        "to_address": "a_4",
        "flow_amount": 32,
        "time": 8,
        "n_inputs": 1,
        "n_outputs": 2,
        "total_amount": 32.003
      },
      {
        "from_address": "a_6",
        "to_address": "a_6",
        "flow_amount": 0.003,
        "time": 8,
        "n_inputs": 1,
        "n_outputs": 2,
        "total_amount": 0.003
      }, 
      {
        "from_address": "a_8",
        "to_address": "a_9",
        "flow_amount": 10,
        "time": 9,
        "n_inputs": 1,
        "n_outputs": 1,
        "total_amount": 10
      },
      {
        "from_address": "a_10",
        "to_address": "a_11",
        "flow_amount": 6,
        "time": 10,
        "n_inputs": 1,
        "n_outputs": 1,
        "total_amount": 6
      },
      {
        "from_address": "a_12",
        "to_address": "a_13",
        "flow_amount": 67,
        "time": 11,
        "n_inputs": 1,
        "n_outputs": 1,
        "total_amount": 67
      }, 
    ]

    # nodo anomalo: super hub
    rows.append({
        "from_address": "a_hub",
        "to_address": "a_14",
        "flow_amount": 1000,
        "time": 12,
        "n_inputs": 1,
        "n_outputs": 50,
        "total_amount": 1000
    })

    print(rows)
    df = pd.DataFrame(rows)
    addresses = sorted(set(df.from_address) | set(df.to_address))
    return df, addresses