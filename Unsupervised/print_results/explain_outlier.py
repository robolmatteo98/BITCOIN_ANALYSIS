def explain_outlier(idx, node_stats, addresses):
  print(f"\n=== SPIEGAZIONE OUTLIER: {addresses[idx]} ===")
  row = node_stats.loc[idx]
  for col in node_stats.columns:
    print(f"{col}: {row[col]}")