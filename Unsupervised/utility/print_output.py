from classification.Anomaly_classification_with_scores import classify_node_with_scores

def print_results(indices_sospetti, df_edges, addresses):
  print("\n=== DETTAGLI NODI SOSPETTI ===")
  for i in indices_sospetti:
    addr = addresses[i]
    reason = classify_node_with_scores(addr, df_edges)
    node_edges = df_edges[(df_edges['from_address'] == addr) | (df_edges['to_address'] == addr)]
    out_edges = node_edges[node_edges['from_address'] == addr]
    in_edges = node_edges[node_edges['to_address'] == addr]
    
    print(f"\nIndirizzo sospetto: {addr}")
    for label, score in reason:
      print(f"  - {label}: {(score * 100):.2f} %")
    print(f"  Grado uscente: {len(out_edges)}, somma uscente: {out_edges['flow_amount'].sum():.2f}")
    print(f"  Grado entrante: {len(in_edges)}, somma entrante: {in_edges['flow_amount'].sum():.2f}")
    print(node_edges[['txid', 'from_address','to_address','flow_amount','time']])
    print("---------------------------------------------------")