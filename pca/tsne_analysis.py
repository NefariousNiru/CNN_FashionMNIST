import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def perform_tsne(train_loader, n_components=2):
    images, labels = next(iter(train_loader))
    images = images.view(images.size(0), -1)  # Flatten the images

    # Perform t-SNE
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_result = tsne.fit_transform(images)

    # Plot t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10')
    plt.title("t-SNE on Fashion MNIST")
    plt.colorbar(scatter, ticks=range(10))
    plt.show()
