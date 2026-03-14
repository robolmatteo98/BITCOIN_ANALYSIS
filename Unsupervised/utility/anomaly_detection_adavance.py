import torch
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def detect_anomalies_latent_advanced(z, k_neighbors=20, quantile=0.99, alpha=0.5):
    """
    z: tensor (num_nodes, latent_dim)
    k_neighbors: numero di vicini per LOF
    quantile: soglia percentile
    alpha: peso tra Mahalanobis e LOF (0.5 = media)
    """

    # Convert to numpy
    Z = z.detach().cpu().numpy()

    # 1) Mahalanobis distance
    mu = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False)
    cov_inv = np.linalg.pinv(cov)

    diff = Z - mu
    mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    # 2) Local Outlier Factor
    lof_model = LocalOutlierFactor(
        n_neighbors=k_neighbors,
        contamination="auto",
        novelty=False
    )
    lof_scores = -lof_model.fit_predict(Z)  # 1 = normale, -1 = outlier
    lof_values = -lof_model.negative_outlier_factor_

    # 3) Combined score
    combined = alpha * mahal + (1 - alpha) * lof_values

    # 4) Threshold via quantile
    threshold = np.quantile(combined, quantile)
    indices = np.where(combined > threshold)[0]

    print("Mahalanobis max:", mahal.max())
    print("LOF max:", lof_values.max())
    print("Combined threshold:", threshold)
    print("Num anomalies:", len(indices))

    return {
        "indices": indices.tolist(),
        "combined_scores": combined,
        "mahalanobis": mahal,
        "lof": lof_values,
        "threshold": threshold
    }