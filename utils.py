from datetime import datetime, timedelta
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import utils


def get_train_test_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return train_transform, test_transform

def load(train_transform, test_transform):
    train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=test_transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def visualize_predictions(model, test_loader, num_images=5):
    model.eval()
    images, labels = next(iter(test_loader))
    class_names = get_class_names()

    images = images.to(next(model.parameters()).device)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        predicted_label = class_names[predictions[i].item()]
        actual_label = class_names[labels[i].item()]
        plt.title(f'Pred: {predicted_label}\nActual: {actual_label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_class_names():
    return ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def plot_confusion_matrix(conf_matrix):
    class_labels = utils.get_class_names()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def get_time():
    return time.time()

def time_string(t):
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

def get_time_diff(start, end):
    return str(timedelta(seconds = end - start))