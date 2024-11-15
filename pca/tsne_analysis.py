import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import utils


def perform_tsne_2d(tsne_result, labels, class_names):
    n_classes = len(class_names)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title("t-SNE 2D Plot")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    cb = plt.colorbar(scatter, ticks=range(n_classes))
    cb.ax.set_yticklabels(class_names)
    plt.show()


def perform_tsne_3d(tsne_result, labels, class_names):
    n_classes = len(class_names)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2],
                         c=labels, cmap='tab10', alpha=0.6)
    ax.set_title("t-SNE 3D Plot")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")

    cb = plt.colorbar(scatter, ticks=range(n_classes))
    cb.ax.set_yticklabels(class_names)
    plt.show()


def perform_tsne(train_loader, n_components):
    class_names = utils.get_class_names()

    images, labels = next(iter(train_loader))

    images = images.view(images.size(0), -1)
    images = StandardScaler().fit_transform(images)

    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, max_iter=300)
    tsne_result = tsne.fit_transform(images)

    if n_components == 2:
        perform_tsne_2d(tsne_result, labels, class_names)
    elif n_components == 3:
        perform_tsne_3d(tsne_result, labels, class_names)
    else:
        print(f"n_components={n_components} is not supported for visualization. Use 2 or 3.")
