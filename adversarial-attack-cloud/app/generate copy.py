import torch
import torchattacks
from torchvision import datasets, transforms
from PIL import Image
import io
import base64
import google.generativeai as genai
import random
import os
import time


from app.models import LeNet5, SqueezeNetMNIST


MODEL_REGISTRY = {
    "lenet": {"class": LeNet5, "path": "app/model_weights/lenet.pth"},
    "lenet_robust": {"class": LeNet5, "path": "app/model_weights/lenet_robust.pth"},
    "squeezenet": {"class": SqueezeNetMNIST, "path": "app/model_weights/squeezenet.pth"},
    "squeezenet_robust": {"class": SqueezeNetMNIST, "path": "app/model_weights/squeezenet_robust.pth"},
}


def load_model(model_name: str):
    config = MODEL_REGISTRY[model_name]
    model = config["class"]()
    model.load_state_dict(torch.load(config["path"], map_location="cpu"))
    model.eval()
    return model


def get_mnist_samples(n: int = 20):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)
    indices = random.sample(range(len(dataset)), n)
    images = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])
    return images, labels


def generate_adversarial_examples(model, images, labels):
    attack = torchattacks.PGD(model, eps=0.3, steps=7, random_start=True)
    adversarial_images = attack(images, labels)
    return adversarial_images


def tensor_to_base64(tensor):
    img_array = (tensor.squeeze().numpy() * 255).astype("uint8")
    img = Image.fromarray(img_array, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def classify_with_gemini(image_base64: str, api_key: str) -> int:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    image_bytes = base64.b64decode(image_base64)
    
    for attempt in range(3):
        try:
            response = model.generate_content([
                "Classify the handwritten digit in this image. Respond with only a single digit 0-9, nothing else.",
                {"mime_type": "image/png", "data": image_bytes}
            ])
            return int(response.text.strip())
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(25)  # Wait for quota reset
            else:
                raise e
    
    return -1


def run_evaluation(model_name: str, api_key: str, num_examples: int = 20):
    model = load_model(model_name)
    images, labels = get_mnist_samples(num_examples)
    adversarial_images = generate_adversarial_examples(model, images, labels)
    
    results = []
    correct = 0
    
    for i in range(len(adversarial_images)):
        img_base64 = tensor_to_base64(adversarial_images[i])
        true_label = labels[i].item()
        predicted = classify_with_gemini(img_base64, api_key)
        
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
            
        results.append({
            "index": i,
            "true_label": true_label,
            "predicted": predicted,
            "correct": is_correct
        })
    
    return {
        "source_model": model_name,
        "vision_model": "gemini-2.0-flash",
        "total_examples": num_examples,
        "correct": correct,
        "accuracy": correct / num_examples,
        "results": results
    }