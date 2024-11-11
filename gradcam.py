import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_gradcam_heatmap(model, images):
    model.eval()
    images.requires_grad = True

    # Forward pass
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Select the predicted class scores for each image
    selected_outputs = outputs.gather(1, predicted.view(-1, 1)).squeeze()

    # Compute gradients with respect to the selected outputs
    grads = torch.autograd.grad(torch.sum(selected_outputs), images)[0]
    heatmaps = torch.mean(grads, dim=1).squeeze().detach().cpu().numpy()

    return heatmaps, predicted


def visualize_gradcam(model, test_loader, num_images=5):
    model.eval()
    images, labels = next(iter(test_loader))

    # Move images to the device
    images = images.to(next(model.parameters()).device)

    # Get Grad-CAM heatmaps
    heatmaps, predictions = get_gradcam_heatmap(model, images)

    # Plot images with Grad-CAM heatmaps
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        # Detach the image tensor from the graph and convert to numpy
        image_np = images[i].cpu().detach().squeeze().numpy()
        heatmap_np = heatmaps[i]

        # Display the image
        plt.imshow(image_np, cmap='gray')
        plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
        plt.title(f'Pred: {predictions[i].item()}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
