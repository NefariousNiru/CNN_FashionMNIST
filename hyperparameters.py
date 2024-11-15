import torch
from sklearn.model_selection import ParameterGrid
from torch import optim

from models.CNN import CNN
from validation import train_test


def hyperparameter_optimization(train_loader, test_loader, criterion):
    param_grid = {
        'lr': [0.01, 0.001, 0.0001],
        'batch_size': [32, 64, 128]
    }

    best_accuracy = 0
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        model = CNN()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        train_test.train(train_loader, optimizer, model, criterion, 30)
        print("\n----------Test------------")
        _, accuracy = train_test.evaluate(model, test_loader)
        print(f"Params: {params}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model

    print(f'Best Params: {best_params}, Accuracy: {best_accuracy:.2f}%')
    print("Saving Best model")
    torch.save(best_model.state_dict(), "fashion_mnist_cnn.pth")