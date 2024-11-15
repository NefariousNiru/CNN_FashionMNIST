import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils


def perform_pca(train_loader, n_components):
    class_names = utils.get_class_names()
    images, labels = next(iter(train_loader))

    images = images.view(images.size(0), -1)
    images = StandardScaler().fit_transform(images)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(images)

    if n_components == 2:
        perform_pca_2d(pca_result, labels, class_names)
    elif n_components == 3:
        perform_pca_3d(pca_result, labels, class_names)
    else:
        print(f"n_components={n_components} is not supported for visualization. Use 2 or 3.")



def perform_pca_2d(pca_result, labels, class_names):
    n_classes = len(class_names)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title("PCA 2D Plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    cb = plt.colorbar(scatter, ticks=range(n_classes))
    cb.ax.set_yticklabels(class_names)
    plt.show()


def perform_pca_3d(pca_result, labels, class_names):
    n_classes = len(class_names)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=labels, cmap='tab10', alpha=0.6)

    ax.set_title("PCA 3D Plot")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    cb = plt.colorbar(scatter, ticks=range(n_classes))
    cb.ax.set_yticklabels(class_names)
    plt.show()
