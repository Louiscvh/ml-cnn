from src.model.convnet import ConvNet
from src.model.mlp import MLP
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI()

# Function to load saved ConvNet model
def load_convnet_model(model_path):
    model = ConvNet(input_size=1, n_kernels=6, output_size=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess image for prediction
def preprocess_image(image):
    # Convert image to grayscale, resize to 28x28, and convert to tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Load ConvNet model
MODEL_PATH = "model/mnist-0.0.1.pt"
convnet_model = load_convnet_model(MODEL_PATH)

@app.post("/api/v1/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Preprocess image for prediction
        input_tensor = preprocess_image(image)
        
        # Make prediction using ConvNet model
        with torch.no_grad():
            output = convnet_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        return {"prediction": predicted_class}
    
    except Exception as e:
        return {"error": str(e)}

# Main function to train models and save ConvNet model
def main():
    input_size = 1  # For grayscale images
    output_size = 10
    n_kernels = 6
    n_hidden = 8

    # Initialize ConvNet and MLP models
    convnet = ConvNet(input_size, n_kernels, output_size)
    mlp = MLP(input_size=28*28, n_hidden=n_hidden, output_size=output_size)

    # Print model parameters
    print(f"ConvNet Parameters={sum(p.numel() for p in convnet.parameters())/1e3}K")
    print(f"MLP Parameters={sum(p.numel() for p in mlp.parameters())/1e3}K")

    # Load datasets
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training with random permutation
    perm = torch.randperm(784)

    print("\nTraining ConvNet with random permutation:")
    convnet.train_net(train_loader, perm=perm, n_epoch=1)
    convnet.test(test_loader, perm=perm)

    print("\nTraining MLP with random permutation:")
    mlp.train_net(train_loader, perm=perm, n_epoch=1)
    mlp.test(test_loader, perm=perm)

    # Save ConvNet model
    convnet_model_path = "model/mnist-0.0.1.pt"
    save_model(convnet, convnet_model_path)

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Calling the main function
if __name__ == "__main__":
    main()
