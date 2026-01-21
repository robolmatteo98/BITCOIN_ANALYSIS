import torch

# k meno aggressivo , invece di 3 provo 4 o 5
def detect_anomalies_latent(z, k=4):
  norms = torch.norm(z - z.mean(dim=0), dim=1)

  median = norms.median()
  mad = torch.median(torch.abs(norms - median))

  threshold = median + k * mad
  mask = norms > threshold

  indices = mask.nonzero(as_tuple=True)[0].tolist()
  scores = norms[indices].tolist()

  print("median:", median.item())
  print("mad:", mad.item())
  print("threshold:", threshold.item())
  print("max distance:", norms.max().item())
  print("num anomalies:", len(indices))

  return indices, scores, norms, threshold