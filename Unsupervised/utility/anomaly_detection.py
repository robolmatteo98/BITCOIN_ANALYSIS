import torch

def detect_anomalies_latent(z, k=3):
  norms = torch.norm(z - z.mean(dim=0), dim=1)

  median = norms.median()
  mad = torch.median(torch.abs(norms - median))

  threshold = median + k * mad
  mask = norms > threshold

  indices = mask.nonzero(as_tuple=True)[0].tolist()
  scores = norms[indices].tolist()

  return indices, scores, norms, threshold