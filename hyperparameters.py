from sklearn.model_selection import ParameterGrid
from torch import optim
from validation import train_test


def hyperparameter_optimization(train_loader, model, criterion):
    param_grid = {
        'lr': [0.01, 0.001, 0.0001],
        'batch_size': [32, 64, 128]
    }

    best_accuracy = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        train_test.train(train_loader, optimizer, model, criterion)
        accuracy = train_test.evaluate(model, train_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f'Best Params: {best_params}, Accuracy: {best_accuracy:.2f}%')
