import torch

# k meno aggressivo , invece di 3 provo 4 o 5
def detect_anomalies_latent(z, k=3):
  center = z.mean(dim=0)
  norms = torch.norm(z - center, dim=1)

  print("\n--- LATENT VECTORS ---")
  for i in range(z.shape[0]):
      print(f"Node {i}: z={z[i].numpy()}, norm={norms[i]:.4f}")

  median = norms.median()
  mad = torch.median(torch.abs(norms - median))

  if mad == 0:
      print("⚠️ MAD = 0 → embeddings collassati")

  threshold = median + k * mad
  indices = (norms > threshold).nonzero(as_tuple=True)[0].tolist()

  return indices, norms, threshold
