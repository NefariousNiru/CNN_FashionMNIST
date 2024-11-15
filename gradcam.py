import torch
import matplotlib.pyplot as plt
import utils


def get_gradcam_heatmap(model, images):
    model.eval()
    images.requires_grad = True

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    selected_outputs = outputs.gather(1, predicted.view(-1, 1)).squeeze()

    grads = torch.autograd.grad(torch.sum(selected_outputs), images)[0]
    heatmaps = torch.mean(grads, dim=1).squeeze().detach().cpu().numpy()

    return heatmaps, predicted


def visualize_gradcam(model, test_loader, num_images=5):
    model.eval()
    images, labels = next(iter(test_loader))

    images = images.to(next(model.parameters()).device)
    labels = labels.to(next(model.parameters()).device)
    class_names = utils.get_class_names()

    heatmaps, predictions = get_gradcam_heatmap(model, images)

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        image_np = images[i].cpu().detach().squeeze().numpy()
        heatmap_np = heatmaps[i]

        plt.imshow(image_np, cmap='gray')
        plt.imshow(heatmap_np, cmap='jet', alpha=0.5)

        predicted_label = class_names[predictions[i].item()]
        true_label = class_names[labels[i].item()]

        plt.title(f'Pred: {predicted_label}\nTrue: {true_label}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
