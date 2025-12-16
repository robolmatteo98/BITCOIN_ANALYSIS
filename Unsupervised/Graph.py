import networkx as nx
import matplotlib.pyplot as plt

def plot_graph_with_anomalies(df_edges, suspicious_indices, addr_to_idx, filename="bitcoin_graph.png"):
    print("Costruisco grafo NetworkX per visualizzazione…")

    # Crea grafo diretto
    G = nx.DiGraph()

    # Aggiunge nodi
    for addr, idx in addr_to_idx.items():
        G.add_node(idx)

    # Aggiunge archi
    for _, row in df_edges.iterrows():
        G.add_edge(row["src"], row["dst"])

    # Colori dei nodi: rosso = anomalo, blu = normale
    node_colors = []
    suspicious_set = set(suspicious_indices)

    for idx in G.nodes():
        if idx in suspicious_set:
            node_colors.append("red")
        else:
            node_colors.append("skyblue")

    print("Calcolo layout (può richiedere qualche secondo)…")
    pos = nx.spring_layout(G, k=0.15, iterations=50)

    plt.figure(figsize=(16, 12))
    nx.draw(
        G,
        pos,
        node_size=30,
        node_color=node_colors,
        edge_color="gray",
        arrows=False,
        alpha=0.6
    )

    plt.title("Bitcoin Transaction Graph – Nodi sospetti evidenziati (rosso)")
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Grafo salvato come: {filename}")