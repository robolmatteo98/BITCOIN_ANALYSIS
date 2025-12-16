import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def analyze_suspicious_nodes(df_edges, addresses, indices_sospetti, addr_to_idx):
    """
    df_edges: dataframe con colonne ['from_address','to_address','flow_amount','time']
    addresses: lista di tutti gli indirizzi
    indices_sospetti: lista di indici dei nodi sospetti nella lista addresses
    addr_to_idx: mappa indirizzo -> indice numerico
    """
    
    # 1. Costruisci il grafo completo
    G = nx.DiGraph()
    for _, row in df_edges.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=row['flow_amount'], time=row['time'])
    
    # 2. Analisi per ciascun nodo sospetto
    for i in indices_sospetti:
        node = addresses[i]
        print(f"\n=== ANALISI NODO SOSPETTO: {node} ===")
        
        # Vicini a 1 hop
        neighbors_in = list(G.predecessors(node))
        neighbors_out = list(G.successors(node))
        
        # Metriche
        in_edges = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))
        
        in_degree = len(in_edges)
        out_degree = len(out_edges)
        sum_in = sum([d['weight'] for _, _, d in in_edges])
        sum_out = sum([d['weight'] for _, _, d in out_edges])
        ratio = (sum_out / sum_in) if sum_in > 0 else float('inf')
        total_tx = in_degree + out_degree
        
        print(f"Grado entrante: {in_degree}, somma flussi entrante: {sum_in:.2f}")
        print(f"Grado uscente: {out_degree}, somma flussi uscente: {sum_out:.2f}")
        print(f"Rapporto uscente/entrante: {ratio:.2f}")
        print(f"Numero totale transazioni: {total_tx}")
        print(f"Vicini entranti: {neighbors_in}")
        print(f"Vicini uscenti: {neighbors_out}")
        
        # 3. Sotto-grafo 1-hop
        sub_nodes = [node] + neighbors_in + neighbors_out
        SG = G.subgraph(sub_nodes)
        
        # Disegna
        plt.figure(figsize=(6,6))
        pos = nx.spring_layout(SG, seed=42)

        # Colori: rosso nodo sospetto, blu gli altri
        node_colors = ['red' if n == node else 'skyblue' for n in SG.nodes()]

        # Etichette: solo prime 3 cifre
        labels = {n: n[:3] for n in SG.nodes()}

        nx.draw(SG, pos, with_labels=True, labels=labels, node_color=node_colors, node_size=800, arrowsize=20)

        # Edge labels (peso)
        edge_labels = {(u,v): f"{d['weight']:.1f}" for u,v,d in SG.edges(data=True)}
        nx.draw_networkx_edge_labels(SG, pos, edge_labels=edge_labels)

        plt.title(f"Sotto-grafo 1-hop nodo sospetto: {node}")
        plt.show()

