import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def perform_pca(train_loader, n_components=2):
    images, labels = next(iter(train_loader))
    images = images.view(images.size(0), -1)  # Flatten the images

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(images)

    # Plot PCA results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10')
    plt.title("PCA on Fashion MNIST")
    plt.colorbar(scatter, ticks=range(10))
    plt.show()
