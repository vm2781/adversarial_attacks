import torch
import torchattacks
import pandas as pd
import numpy as np
import random
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision import transforms
from PIL import Image
from fastapi import HTTPException
import os
import io
import base64
import google.generativeai as genai
import time


# ImageNette class mapping
IMAGENETTE_CLASSES = {
    'n01440764': 'tench',
    'n02102040': 'English springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
}

IMAGENETTE_ID_TO_IDX = {k: i for i, k in enumerate(IMAGENETTE_CLASSES.keys())}
IMAGENETTE_IDX_TO_NAME = {i: v for i, (k, v) in enumerate(IMAGENETTE_CLASSES.items())}

# ImageNet class indices for ImageNette classes (for pretrained SqueezeNet)
IMAGENET_CLASS_INDICES = {
    'n01440764': 0,
    'n02102040': 217,
    'n02979186': 482,
    'n03000684': 491,
    'n03028079': 497,
    'n03394916': 566,
    'n03417042': 569,
    'n03425413': 571,
    'n03445777': 574,
    'n03888257': 701
}


def load_imagenette_model():
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def load_imagenette_data(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    base_path = "/app/app/imagenette"
    csv_path = os.path.join(base_path, "noisy_imagenette.csv")
    
    df = pd.read_csv(csv_path)
    val_df = df[df['is_valid'] == True]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images, labels, imagenet_labels = [], [], []
    
    for _, row in val_df.iterrows():
        img_path = os.path.join(base_path, row['path'])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
            
            wnid = row['noisy_labels_0']
            labels.append(IMAGENETTE_ID_TO_IDX[wnid])
            imagenet_labels.append(IMAGENET_CLASS_INDICES[wnid])
    
    # Convert lists to tensors
    images = torch.stack(images)
    labels = torch.tensor(labels)
    imagenet_labels = torch.tensor(imagenet_labels)

    # Shuffle deterministically
    idxs = torch.randperm(len(images))
    images = images[idxs]
    labels = labels[idxs]
    imagenet_labels = imagenet_labels[idxs]
    
    return images, labels, imagenet_labels



def tensor_to_base64(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = Image.fromarray(img_array, mode="RGB")
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def generate_adversarial_examples(model, attack_type, images, labels):
    if attack_type == "pgd":
        attack = torchattacks.PGD(model, eps=0.6, steps=20)
    elif attack_type == "mifgsm":
        attack = torchattacks.MIFGSM(model, eps=0.6, steps=20)
    elif attack_type == "pixle":
        attack = torchattacks.Pixle(model, restarts=30, max_iterations=20)
    else:
        raise HTTPException(
            status_code=400,
            detail="attack specified wasn't an option"
        )
    return attack(images, labels)



def classify_with_gemini(image_base64: str, api_key: str) -> int:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    image_bytes = base64.b64decode(image_base64)
    
    prompt = """Classify this image into exactly one of these 10 categories:
0: tench (a type of fish)
1: English springer (a dog breed)
2: cassette player
3: chain saw
4: church
5: French horn
6: garbage truck
7: gas pump
8: golf ball
9: parachute

Respond with ONLY a single digit 0-9, nothing else."""
    
    for attempt in range(3):
        try:
            response = model.generate_content(
                [prompt, {"mime_type": "image/png", "data": image_bytes}],
                generation_config={"temperature": 0}
            )
            return int(response.text.strip())
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(25)
            else:
                raise e
    return -1


def run_evaluation(attack_type: str, api_key: str, seed: int, num_examples: int = 20):
    model = load_imagenette_model()
    all_images, all_labels, all_imagenet_labels = load_imagenette_data(seed=seed)
    
    valid_originals = []
    valid_adversarials = []
    valid_labels = []
    idx = 0
    
    while len(valid_adversarials) < num_examples and idx < len(all_images):
        batch_size = min(50, len(all_images) - idx)
        images = all_images[idx:idx + batch_size]
        labels = all_labels[idx:idx + batch_size]
        imagenet_labels = all_imagenet_labels[idx:idx + batch_size]
        idx += batch_size
        
        adversarial_images = generate_adversarial_examples(model, attack_type, images, imagenet_labels)
        
        with torch.no_grad():
            outputs = model(adversarial_images)
            predictions = outputs.argmax(dim=1)
        
        for i in range(len(adversarial_images)):
            if len(valid_adversarials) >= num_examples:
                break
            
            true_label = labels[i].item()
            imagenet_label = imagenet_labels[i].item()
            
            # Condition 1: Fools SqueezeNet
            if predictions[i].item() == imagenet_label:
                continue
            
            # Condition 2: Gemini gets original correct
            orig_base64 = tensor_to_base64(images[i])
            gemini_pred = classify_with_gemini(orig_base64, api_key)
            if gemini_pred != true_label:
                continue
            
            valid_originals.append(images[i])
            valid_adversarials.append(adversarial_images[i])
            valid_labels.append(true_label)
    
    results = []
    correct = 0
    
    for i in range(len(valid_adversarials)):
        true_label = valid_labels[i]
        adv_base64 = tensor_to_base64(valid_adversarials[i])
        predicted = classify_with_gemini(adv_base64, api_key)
        
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        results.append({
            "index": i,
            "true_label": true_label,
            "true_class": IMAGENETTE_IDX_TO_NAME[true_label],
            "predicted": predicted,
            "predicted_class": IMAGENETTE_IDX_TO_NAME.get(predicted, "unknown"),
            "correct": is_correct
        })
    
    return {
        "dataset": "imagenette",
        "source_model": "squeezenet1_1_pretrained",
        "attack_type": attack_type,
        "vision_model": "gemini-2.0-flash",
        "total_tested": len(valid_adversarials),
        "correct": correct,
        "accuracy": correct / len(valid_adversarials) if valid_adversarials else 0,
        "results": results
    }