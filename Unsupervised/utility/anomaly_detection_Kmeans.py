import torch
from sklearn.cluster import KMeans


def detect_anomalies_cluster_distance(z, k=3, n_clusters=6, eps=1e-6):
    # 1. Standardizzazione del latent space
    mean = z.mean(dim=0)
    std = z.std(dim=0) + eps
    z_norm = (z - mean) / std

    # 2. Clustering (KMeans)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(z_norm.cpu().numpy())
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # 3. Distanza dal centroide più vicino
    dists = []
    for i in range(z_norm.shape[0]):
        zi = z_norm[i]
        dist_to_centers = torch.norm(centers - zi, dim=1)
        dists.append(torch.min(dist_to_centers).item())

    dists = torch.tensor(dists)

    # 4. Threshold con MAD
    median = dists.median()
    mad = torch.median(torch.abs(dists - median))

    threshold = median + k * mad
    mask = dists > threshold

    indices = mask.nonzero(as_tuple=True)[0].tolist()
    scores = dists[indices].tolist()

    return indices, scores, dists, threshold
