import torch
import torch.nn.functional as F

def train_model(model, data):
  # ottimizzatore Adam
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
  )

  # ciclo di addestramento
  for epoch in range(200):

    # clear dei gradienti
    optimizer.zero_grad()

    # do al modello scelto i dati
    out = model(data.x, data.edge_index)

    # calcolo l'errore, confrontando la previsione del modello 'out' con la risposta vera e propria 'data.y'
    loss = F.cross_entropy(
      out[data.train_mask],
      data.y[data.train_mask]
    )

    loss.backward() # calcola quanto ogni parte del modello ha sbagliato
    optimizer.step() # sistema i pesi

  return model