import torch
import utils
from metrics import (
    calculate_accuracy, calculate_precision, calculate_recall,
    calculate_f1_score, generate_confusion_matrix
)

def train(train_loader, optimizer, model, criterion, num_epochs=10):
    num_classes = len(utils.get_class_names())
    for epoch in range(num_epochs):
        start = utils.get_time()
        running_loss = 0.0
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_batches = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            accuracy = calculate_accuracy(outputs, labels)
            precision = calculate_precision(outputs, labels, num_classes)
            recall = calculate_recall(outputs, labels, num_classes)
            f1 = calculate_f1_score(outputs, labels, num_classes)

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_batches += 1

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = 100 * total_accuracy / total_batches
        avg_precision = 100 * total_precision / total_batches
        avg_recall = 100 * total_recall / total_batches
        avg_f1 = 100 * total_f1 / total_batches

        end = utils.get_time()
        print(f'\nEpoch [{epoch + 1}/{num_epochs}] '
              f'\nLoss: {avg_loss:.4f} '
              f'\nAccuracy: {avg_accuracy:.2f} % '
              f'\nPrecision: {avg_precision:.2f} % '
              f'\nRecall: {avg_recall:.2f} % '
              f'\nF1 Score: {avg_f1:.2f} %'
              f'\nTook: {utils.get_time_diff(start, end)}')


def evaluate(model, data_loader):
    model.eval()
    class_labels = utils.get_class_names()
    num_classes = len(class_labels)

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_batches = 0
    confusion_matrix_total = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)

            batch_accuracy = calculate_accuracy(outputs, labels)
            batch_precision = calculate_precision(outputs, labels, num_classes)
            batch_recall = calculate_recall(outputs, labels, num_classes)
            batch_f1 = calculate_f1_score(outputs, labels, num_classes)
            batch_conf_matrix = generate_confusion_matrix(outputs, labels, num_classes)

            total_accuracy += batch_accuracy
            total_precision += batch_precision
            total_recall += batch_recall
            total_f1 += batch_f1
            confusion_matrix_total += batch_conf_matrix
            num_batches += 1

    average_accuracy = 100 * total_accuracy / num_batches
    average_precision = 100 * total_precision / num_batches
    average_recall = 100 * total_recall / num_batches
    average_f1 = 100 * total_f1 / num_batches

    print(f"Accuracy: {average_accuracy:.2f}%")
    print(f"Precision: {average_precision:.2f}%")
    print(f"Recall: {average_recall:.2f}%")
    print(f"F1 Score: {average_f1:.2f}%")
    return confusion_matrix_total, average_accuracy