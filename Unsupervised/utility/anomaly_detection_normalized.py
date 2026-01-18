import torch

def detect_anomalies_latent_normalized(z, k=3, eps=1e-6):
  # Standardizzazione del latent space
  mean = z.mean(dim=0)
  std = z.std(dim=0) + eps
  z_norm = (z - mean) / std

  # Distanza normalizzata
  norms = torch.norm(z_norm, dim=1)

  # MAD threshold
  median = norms.median()
  mad = torch.median(torch.abs(norms - median))

  threshold = median + k * mad
  mask = norms > threshold

  indices = mask.nonzero(as_tuple=True)[0].tolist()
  scores = norms[indices].tolist()

  return indices, scores, norms, threshold
