"""Anomaly detection temporale su un multigrafo di transazioni Bitcoin."""
try:
    from .gae_autoencoder.analysis import score_observed_transactions
    from .gae_autoencoder.config import TIME_CUTOFF, set_seed
    from .gae_autoencoder.data import prepare_graph
    from .gae_autoencoder.training import train_model
except ImportError:
    from gae_autoencoder.analysis import score_observed_transactions
    from gae_autoencoder.config import TIME_CUTOFF, set_seed
    from gae_autoencoder.data import prepare_graph
    from gae_autoencoder.training import train_model

def main() -> None:
    print("\n" + "=" * 72)
    print("ANOMALY DETECTION TEMPORALE SU TRANSAZIONI MULTI-EDGE")
    print("=" * 72)
    print(f"Training: timestamp <= {TIME_CUTOFF} | Scoring: timestamp > {TIME_CUTOFF}")
    print("Ogni transazione resta distinta, anche per la stessa coppia di wallet.")
    set_seed(42)

    # Creo il grafo
    data = prepare_graph(time_cutoff=TIME_CUTOFF, seed=42)
    model = train_model(data)
    print(f"\nTransazioni di training: {len(data.train_transactions)}")
    print(f"Transazioni da controllare: {len(data.observed_transactions)}")
    print("\nTransazioni future valutate:")
    results = score_observed_transactions(model, data)
    for src, dst, amount, timestamp, probability, score, _ in results:
        print(
            f"  t={timestamp} | {src} -> {dst} | BTC: {amount:.2f} | "
            f"Probabilità: {probability:.6f} | Anomaly score: {score:.6f}"
        )

    anomalous = [result for result in results if result[-1] == "anomalia probabile"]
    print("\nArchi con anomalia probabile:")
    if not anomalous:
        print("  Nessuno")
    else:
        for src, dst, amount, timestamp, probability, score, _ in anomalous:
            print(
                f"  t={timestamp} | {src} -> {dst} | BTC: {amount:.2f} | "
                f"Probabilità: {probability:.6f} | Anomaly score: {score:.6f}"
            )

if __name__ == "__main__":
    main()
