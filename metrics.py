import torch

def calculate_accuracy(predictions, labels):
    _, predicted_labels = torch.max(predictions, 1)
    correct = (predicted_labels == labels).sum().item()
    return correct / labels.size(0)

def calculate_precision(predictions, labels, num_classes):
    _, predicted_labels = torch.max(predictions, 1)
    precision_per_class = []
    for cls in range(num_classes):
        true_positives = ((predicted_labels == cls) & (labels == cls)).sum().item()
        predicted_positives = (predicted_labels == cls).sum().item()
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        precision_per_class.append(precision)
    return sum(precision_per_class) / num_classes

def calculate_recall(predictions, labels, num_classes):
    _, predicted_labels = torch.max(predictions, 1)
    recall_per_class = []
    for cls in range(num_classes):
        true_positives = ((predicted_labels == cls) & (labels == cls)).sum().item()
        actual_positives = (labels == cls).sum().item()
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        recall_per_class.append(recall)
    return sum(recall_per_class) / num_classes

def calculate_f1_score(predictions, labels, num_classes):
    precision = calculate_precision(predictions, labels, num_classes)
    recall = calculate_recall(predictions, labels, num_classes)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def generate_confusion_matrix(predictions, labels, num_classes):
    _, predicted_labels = torch.max(predictions, 1)
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for actual, predicted in zip(labels.view(-1), predicted_labels.view(-1)):
        conf_matrix[actual.long(), predicted.long()] += 1
    return conf_matrix
