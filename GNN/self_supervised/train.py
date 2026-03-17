import torch
import torch.nn.functional as F

# funzione di separazione train e test a seconda se si trova sotto o sopra la media delle transazioni nella quale è coinvolto quell'indirizzo
def temporal_edge_split(data, cutoff):
    train_mask = data.edge_time < cutoff
    test_mask = data.edge_time >= cutoff

    train_edges = data.edge_index[:, train_mask]
    test_edges = data.edge_index[:, test_mask]

    return train_edges, test_edges


def train_model(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    # split temporale (mediana come cutoff)
    cutoff = data.edge_time.median()
    train_edges, _ = temporal_edge_split(data, cutoff)

    for _ in range(200):
        optimizer.zero_grad() # azzera i gradienti

        # z è un embedding, ovvero un vettore contenente tutte le caratteristiche del nodo nello spazio latente
        z = model(data.x, train_edges)

        # positive edges --> se due nodi sono connessi, il loro prodotto scalare sarà alto; se non lo sono, basso
        pos_src = train_edges[0]
        pos_dst = train_edges[1]
        pos_score = (z[pos_src] * z[pos_dst]).sum(dim=1)

        # negative sampling --> per ogni edge positivo viene generato un edge negativo casuale (nodi non connessi), così vengono soddisfatti anche gli esempi negativi
        num_nodes = data.num_nodes
        neg_src = torch.randint(0, num_nodes, pos_src.size())
        neg_dst = torch.randint(0, num_nodes, pos_dst.size())
        neg_score = (z[neg_src] * z[neg_dst]).sum(dim=1)

        # loss BCE
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score)
            ])
        )

        loss.backward() # calcola i gradienti della loss rispetto ai pesi del modello
        optimizer.step() # aggiorna i pesi secondo Adam usando i gradienti calcolati

    return model


def evaluate_model(model, data):
    model.eval()

    cutoff = data.edge_time.median()
    _, test_edges = temporal_edge_split(data, cutoff)

    with torch.no_grad():
        z = model(data.x, data.edge_index)

        src = test_edges[0]
        dst = test_edges[1]

        # normalizzo lo score del modello
        score = F.sigmoid((z[src] * z[dst]).sum(dim=1))

        return score.mean().item()