import csv
from pathlib import Path

def write_output(args, top, ranks, raw_edge_count, edges_rdd, output_dir, split,
                 transaction_limit, transaction_offset, wallet_mapping, db_stats):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_top_csv(output_dir / f"top_{args.mode}_main.csv", top)
    write_all_scores_csv(output_dir / f"pagerank_{args.mode}_all.csv", ranks)

    summary = {
        "mode": args.mode,
        "source": "postgresql",
        "split": split,
        "aggregate_edges": 1,
        "iterations": args.iterations,
        "top_n": 20,
        "raw_edge_count": raw_edge_count,
        "edge_count": edges_rdd.count(),
        "node_count": edges_rdd.flatMap(lambda edge: [edge["src"], edge["dst"]]).distinct().count(),
    }
    summary["transaction_limit"] = transaction_limit
    summary["transaction_offset"] = transaction_offset
    summary["wallet_mapping"] = str(wallet_mapping) if wallet_mapping else None
    summary.update(db_stats)
    summary_path = output_dir / f"summary_{args.mode}_main.txt"
    write_summary_on_file(summary_path, summary)
    print(summary_path.read_text())

def write_top_csv(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", "score"])
        for node, score in rows:
            writer.writerow([node, score])


def write_all_scores_csv(path, ranks):
    """Scrive il punteggio di ogni nodo senza accumulare tutto nel driver."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", "score"])
        for node, score in ranks.sortByKey().toLocalIterator():
            writer.writerow([node, score])

def write_summary_on_file(path, summary):
    lines = []
    for key, value in summary.items():
        if key == "top_n" and isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)) and len(value[0]) == 2:
            lines.append("top_n:")
            for node, score in value:
                lines.append(f"  - {node},{score}")
        else:
            lines.append(f"{key}: {value}")
    Path(path).write_text("\n".join(lines) + "\n")
