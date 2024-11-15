import time
from torch import nn, optim
import gradcam
import hyperparameters
import utils
from models.CNN import CNN
from pca import pca_analysis, tsne_analysis
from validation import train_test

def log_time(message1, message2):
    print(f"{message1} {message2}")

def run_fashion_mnist():
    start_time = utils.get_time()
    log_time("Start Time:", utils.time_string(start_time))

    train_transform, test_transform = utils.get_train_test_transforms()
    train_loader, test_loader = utils.load(train_transform, test_transform)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("----------Training Start-----------")
    train_test.train(train_loader, optimizer, model, criterion, 20)
    training_end_time = utils.get_time()
    log_time("\nTraining Finished:", utils.time_string(training_end_time))
    log_time("Took:", utils.get_time_diff(start_time, training_end_time))
    print("\n----------Training End-----------\n")


    print("---------Test Start-----------")
    confusion_matrix, _ = train_test.evaluate(model, test_loader)
    print("---------Test End-----------")

    # Plots
    print("\nPlot Confusion Matrix")
    utils.plot_confusion_matrix(confusion_matrix)
    print("\nVisualize Predictions")
    utils.visualize_predictions(model, test_loader)

    print("\nRunning PCA Analysis...")
    pca_analysis.perform_pca(train_loader, 2)
    pca_analysis.perform_pca(train_loader, 3)

    print("\nRunning t-SNE Analysis...")
    tsne_analysis.perform_tsne(train_loader, 2)
    tsne_analysis.perform_tsne(train_loader, 3)

    # Grad-CAM
    print("\nVisualizing Feature Importance using Grad-CAM...")
    gradcam.visualize_gradcam(model, test_loader)

    print("\nRunning Hyperparameter Optimization...")
    hyperparameters.hyperparameter_optimization(train_loader, test_loader, criterion)


if __name__ == '__main__':
    run_fashion_mnist()