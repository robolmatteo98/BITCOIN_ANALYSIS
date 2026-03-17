import torch
import torch.nn.functional as F

def train_model(model, data):
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
  )

  for epoch in range(200):

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = F.cross_entropy(
      out[data.train_mask],
      data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    return model