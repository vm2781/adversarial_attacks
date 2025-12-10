import torch
import torchattacks
from torchvision import datasets, transforms
from fastapi import HTTPException
from PIL import Image
import io
import base64
import google.generativeai as genai
import random
import time

from app.models import LeNet5, SqueezeNetMNIST


MODEL_REGISTRY = {
    "lenet": {"class": LeNet5, "path": "app/model_weights/lenet.pth"},
    "squeezenet": {"class": SqueezeNetMNIST, "path": "app/model_weights/squeezenet.pth"},
    "lenet_pgd": {"class": LeNet5, "path": "app/model_weights/lenet_pgd.pth"},
    "squeezenet_pgd": {"class": SqueezeNetMNIST, "path": "app/model_weights/squeezenet_pgd.pth"},
    "lenet_mifgsm": {"class": LeNet5, "path": "app/model_weights/lenet_mifgsm.pth"},
    "squeezenet_mifgsm": {"class": SqueezeNetMNIST, "path": "app/model_weights/squeezenet_mifgsm.pth"},
    "lenet_pixle": {"class": LeNet5, "path": "app/model_weights/lenet_pixle.pth"},
    "squeezenet_pixle": {"class": SqueezeNetMNIST, "path": "app/model_weights/squeezenet_pixle.pth"},
}


def load_model(model_name: str):
    config = MODEL_REGISTRY[model_name]
    model = config["class"]()
    model.load_state_dict(torch.load(config["path"], map_location="cpu"))
    model.eval()
    return model



def tensor_to_base64(tensor):
    img_array = (tensor.squeeze().numpy() * 255).astype("uint8")
    img = Image.fromarray(img_array, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def generate_adversarial_examples(model, attack_type, images, labels):
    if attack_type == "pgd":
        attack = torchattacks.PGD(model, eps=0.3, steps=7, random_start=True)
    elif attack_type == "mifgsm":
        attack = torchattacks.MIFGSM(model, eps=0.3, steps=7, decay=1.0)
    elif attack_type == "pixle":
        attack = torchattacks.Pixle(model, restarts=10, max_iterations=5)
    else:
        raise HTTPException(
            status_code=400,
            detail="attack specified wasn't an option"
        )
    return attack(images, labels)


def classify_original_with_gemini(image_base64: str, api_key: str) -> int:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    image_bytes = base64.b64decode(image_base64)
    
    for attempt in range(3):
        try:
            response = model.generate_content(
                [
                    "Classify the handwritten digit in this image. Respond with only a single digit 0-9, nothing else.",
                    {"mime_type": "image/png", "data": image_bytes}
                ],
                generation_config={"temperature": 0}
            )
            return int(response.text.strip())
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(25)
            else:
                raise e
    return -1


def classify_adversarial_with_gemini(image_base64: str, api_key: str) -> int:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    image_bytes = base64.b64decode(image_base64)
    
    for attempt in range(3):
        try:
            response = model.generate_content(
                [
                    "Classify the handwritten digit in this image. Respond with only a single digit 0-9, nothing else.",
                    {"mime_type": "image/png", "data": image_bytes}
                ],
                generation_config={"temperature": 0}
            )
            return int(response.text.strip())
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(25)
            else:
                raise e
    return -1


def run_evaluation(model_name: str, attack_type: str, api_key: str, num_examples: int = 20):
    model = load_model(model_name)
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)
    
    valid_originals = []
    valid_adversarials = []
    valid_labels = []
    idx = 0
    
    # Collect num_examples where: (1) fools source model AND (2) Gemini gets original correct
    while len(valid_adversarials) < num_examples and idx < len(dataset):
        batch_size = min(50, len(dataset) - idx)
        images = torch.stack([dataset[i][0] for i in range(idx, idx + batch_size)])
        labels = torch.tensor([dataset[i][1] for i in range(idx, idx + batch_size)])
        idx += batch_size
        
        adversarial_images = generate_adversarial_examples(model, attack_type, images, labels)
        
        # Check which fool the source model
        with torch.no_grad():
            outputs = model(adversarial_images)
            predictions = outputs.argmax(dim=1)
        
        for i in range(len(adversarial_images)):
            if len(valid_adversarials) >= num_examples:
                break
            
            true_label = labels[i].item()
            
            # Condition 1: Fools source model
            if predictions[i].item() == true_label:
                continue
            
            # Condition 2: Gemini gets original correct
            orig_base64 = tensor_to_base64(images[i])
            gemini_pred = classify_original_with_gemini(orig_base64, api_key)
            if gemini_pred != true_label:
                continue
            
            # Both conditions met
            valid_originals.append(images[i])
            valid_adversarials.append(adversarial_images[i])
            valid_labels.append(true_label)
    
    # Now classify all valid adversarials with Gemini
    results = []
    correct = 0
    
    for i in range(len(valid_adversarials)):
        true_label = valid_labels[i]
        adv_base64 = tensor_to_base64(valid_adversarials[i])
        predicted = classify_adversarial_with_gemini(adv_base64, api_key)
        
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
        "attack_type": attack_type,
        "vision_model": "gemini-2.0-flash",
        "total_examples": len(valid_adversarials),
        "correct": correct,
        "accuracy": correct / len(valid_adversarials) if valid_adversarials else 0,
        "results": results
    }