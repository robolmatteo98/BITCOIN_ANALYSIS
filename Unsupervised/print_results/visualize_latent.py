import umap
import matplotlib.pyplot as plt
import numpy as np

def visualize_latent_space(z, indices_sospetti, addresses, output_path="latent_space.png"):
    z_np = z.detach().cpu().numpy()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    z_2d = reducer.fit_transform(z_np)

    colors = np.array(["blue"] * len(z_np))
    for idx in indices_sospetti:
        colors[idx] = "red"

    plt.figure(figsize=(12, 10))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, s=10, alpha=0.7)

    # Etichette SOLO per gli outlier
    for idx in indices_sospetti:
        x, y = z_2d[idx]
        plt.text(x, y, addresses[idx][:6], fontsize=6, color="red")

    plt.title("Latent Space Visualization (UMAP)")
    plt.savefig(output_path, dpi=300)
    plt.close()

    return z_2d