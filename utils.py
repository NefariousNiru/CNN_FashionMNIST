import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform

def load(transform):
    train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    # Create data loaders
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

    # Plot images with predictions
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