from torch import nn, optim

import gradcam
import hyperparameters
import utils
from models.CNN import CNN
from pca import pca_analysis, tsne_analysis
from validation import train_test


def run_fashion_mnist():
    transform = utils.get_transforms()
    train_loader, test_loader = utils.load(transform)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train & Evaluate
    train_test.train(train_loader, optimizer, model, criterion, 1)
    accuracy = train_test.evaluate(model, test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Visualise predictions
    utils.visualize_predictions(model, test_loader)

    # Dimensionality Reduction
    print("\nRunning PCA Analysis...")
    pca_analysis.perform_pca(train_loader)

    print("\nRunning t-SNE Analysis...")
    tsne_analysis.perform_tsne(train_loader)

    # Grad-CAM
    print("\nVisualizing Feature Importance using Grad-CAM...")
    gradcam.visualize_gradcam(model, test_loader)

    print("\nRunning Hyperparameter Optimization...")
    hyperparameters.hyperparameter_optimization(train_loader, model, criterion)


if __name__ == '__main__':
    run_fashion_mnist()