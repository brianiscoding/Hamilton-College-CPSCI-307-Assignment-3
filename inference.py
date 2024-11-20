import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# Assuming we're working with a pre-trained PyTorch model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Example of a simple model (adjust for your use case)
        self.fc1 = nn.Linear(128, 10)  # Modify to match your model architecture

    def forward(self, x):
        return self.fc1(x)


def load_model(model_path):
    """
    Loads the pre-trained model from the file
    """
    model = MyModel()
    model.load_state_dict(torch.load(model_path))  # Load model weights
    model.eval()  # Set to evaluation mode
    return model


def preprocess_image(image_path):
    """
    Preprocesses the image to fit the model input requirements.
    Modify this according to your model's input format.
    """
    # Define the necessary transformations (resize, normalization, etc.)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Example size, modify as needed
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Example for pretrained models
        ]
    )

    # Open the image
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)
    return image


def predict(model, image_tensor):
    """
    Makes a prediction with the model
    """
    with torch.no_grad():  # Turn off gradient computation for inference
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()


def main():
    model_path = "path_to_trained_model.pth"  # Path to your model file
    image_path = "path_to_image.jpg"  # Path to the image to be predicted

    # Step 1: Load the trained model
    model = load_model(model_path)

    # Step 2: Preprocess the input image
    image_tensor = preprocess_image(image_path)

    # Step 3: Predict using the model
    predicted_class = predict(model, image_tensor)

    # Output the prediction result
    print(f"Predicted Class: {predicted_class}")


if __name__ == "__main__":
    main()
