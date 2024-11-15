import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys

import utils
from models.CNN import CNN

def predict(imgPath, modelPath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    model.load_state_dict(torch.load(modelPath, map_location=device, weights_only=True))
    model.eval()

    # Step 2: Define image preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure the image is in grayscale
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        image = Image.open(imgPath)
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    class_names = utils.get_class_names()
    print(f"Predicted class: {class_names[predicted_class]}")

if __name__ == "__main__":
    imgPath = "images/tshirt.png"  # Specify the image path here
    modelPath = "../fashion_mnist_cnn.pth"  # Specify the model path here
    predict(imgPath, modelPath)
