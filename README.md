# Adversarial Attacks on MNIST

This repository contains code, experiments, and analysis for studying **adversarial attacks and robustness** of neural networks on the MNIST dataset. We evaluate **white-box and black-box adversarial attacks**, train **robust models**, and analyze **transferability, robustness, and perturbation magnitudes** across a diverse zoo of architectures, including real-world transfer to **Google Gemini 2.0 Flash**.

---

## Overview

Neural networks achieve impressive accuracy on image classification tasks, yet remain vulnerable to **adversarial examples** (inputs with small, often imperceptible perturbations that cause confident misclassification). This project provides a comprehensive study of:

- **Attack effectiveness**: Comparing PGD, MI-FGSM, and Pixle attacks
- **Defensive robustness**: Evaluating standard vs. adversarially-trained models
- **Transferability**: How well do adversarial examples crafted on one model fool others?
- **Cloud transfer attacks**: Testing adversarial examples against commercial vision APIs (Gemini 2.0 Flash)
- **Perturbation analysis**: Quantifying the L₂ distortion of different attacks

---

## Key Findings

| Finding | Details |
|---------|---------|
| **Robust models generate more transferable attacks** | Adversarially-trained models produce perturbations that transfer better to other architectures, including commercial cloud models |
| **Pixle dominates on small images** | On 28×28 MNIST images, Pixle achieves the highest transfer success (up to 56% attack success on Gemini) |
| **MI-FGSM excels on larger images** | On 320×320 Imagenette images, MI-FGSM outperforms other attacks (9.5% vs 6.5% for PGD, 0.5% for Pixle) |
| **Smaller models can generate stronger transfer attacks** | LeNet variants often outperformed larger SqueezeNet in generating transferable adversarial examples |
| **Perturbation size ≠ attack strength** | MI-FGSM uses the smallest perturbations but isn't always the most effective |

---

## Attacks Implemented

### 1. PGD (Projected Gradient Descent)

A strong iterative white-box attack that repeatedly steps in the direction of the loss gradient while projecting back onto an ε-ball:

$$x^{t+1} = \Pi_{x+S}\left(x^t + \alpha \cdot \text{sign}(\nabla_x L(\theta, x^t, y))\right)$$

**Parameters**: `eps=0.3`, `steps=20`

### 2. MI-FGSM (Momentum Iterative FGSM)

Extends FGSM with momentum to stabilize gradient updates and improve transferability (Dong et al., 2018):

$$g^{t+1} = \mu g^t + \frac{\nabla_x L(\theta, x^t, y)}{\|\nabla_x L(\theta, x^t, y)\|_1}$$

$$x^{t+1} = x^t + \alpha \cdot \text{sign}(g^{t+1})$$

**Parameters**: `eps=0.3`, `steps=20`, `decay=1.0`

### 3. Pixle

A black-box attack that perturbs images by rearranging pixels through random search—no gradient access required (Pomponi et al., 2022):

**Parameters**: `restarts=10`, `max_iterations=5`

---

## Models

### Core Models (for Cloud Transfer Experiments)

| Model | Parameters | Training Variants |
|-------|-----------|-------------------|
| **LeNet-5** | ~44k | Standard, PGD-robust, MI-FGSM-robust, Pixle-robust |
| **SqueezeNet** | ~740k | Standard, PGD-robust, MI-FGSM-robust, Pixle-robust |

A copy of the trained model weights can be found under [`cloud_attack/app/model_weights/`](cloud_attack/app/model_weights/).

### Model Zoo (for Local Transferability Analysis)

- Linear Classifier
- MLP (Multi-Layer Perceptron)
- ConvNet variants (Tiny, Wide, Deep)
- LeNet-5
- SqueezeNet (MNIST-adapted)
- ResNet-18 (MNIST-adapted)
- Mini-VGG, Mini-Inception, Mini-DenseNet

---

## Robust Training

Robust models are trained using **adversarial augmentation**:

```python
# For each batch during training:
# 1. Generate adversarial examples on-the-fly
adv_images = attack(images, labels)

# 2. Train on both clean and adversarial examples
outputs = model(torch.cat([images, adv_images]))
loss = criterion(outputs, torch.cat([labels, labels]))
```

This effectively doubles the training set and forces models to learn features robust to the specific attack type.

---

## Results

### Cloud Transfer Attack Success (MNIST → Gemini 2.0 Flash)


| Source Model | PGD | MI-FGSM | Pixle |
|-------------|-----|---------|-------|
| LeNet (Standard) | 6% | 3% | 38% |
| LeNet (PGD-robust) | 13% | 14% | **56%** |
| LeNet (MI-FGSM-robust) | 16% | **20%** | 51% |
| SqueezeNet (Standard) | 5% | 6% | 12% |
| SqueezeNet (PGD-robust) | 13% | 5% | 32% |

*Attack success = percentage of adversarial examples that fooled Gemini*

### Perturbation Magnitude (L₂ Distance)


| Attack | Avg L₂ Perturbation |
|--------|---------------------|
| MI-FGSM | 2.8 - 6.1 |
| PGD | 4.1 - 5.5 |
| Pixle | 5.7 - 7.6 |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vm2781/adversarial_attacks.git
cd adversarial_attacks

# Install dependencies
pip install torch torchvision torchattacks numpy matplotlib seaborn tqdm
```

### Train Models

```bash
# Train standard LeNet
python src/train_model.py --model lenet --epochs 10

# Train PGD-robust SqueezeNet
python src/train_model.py --model squeezenet --robust --attack pgd --epochs 10

# Train MI-FGSM-robust LeNet
python src/train_model.py --model lenet --robust --attack mifgsm --epochs 10
```

### Run Notebooks

Open any notebook in `notebooks/` to reproduce experiments:

```bash
jupyter notebook notebooks/adversarial_analysis.ipynb
```

Or use Google Colab (all notebooks include Colab badges for one-click execution).

---

## Cloud Experiments

For testing adversarial examples against Google Gemini 2.0 Flash, see the detailed instructions in:

📖 **[cloud_attack/README.md](cloud_attack/CloudREADME.md)**

The cloud pipeline:
1. Deploys trained models to Google Cloud Run
2. Generates adversarial examples on-demand via FastAPI
3. Sends examples to Gemini 2.0 Flash for classification
4. Records transfer attack success rates

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Clean Accuracy** | Accuracy on unperturbed test images |
| **White-box Accuracy** | Accuracy on adversarial examples crafted against the same model |
| **Transfer Success** | Percentage of adversarial examples that fool a different model |
| **L₂ Perturbation** | Average Euclidean distance between clean and adversarial images |

---

## Limitations

1. **Budget mismatch across attacks**: PGD/MI-FGSM use ε-bounded L∞ perturbations, while Pixle uses unbounded pixel rearrangements
2. **Dataset simplicity**: MNIST's small 28×28 images may not generalize to real-world scenarios (partially addressed with Imagenette experiments)
3. **Limited robust training methods**: Only on-the-fly adversarial augmentation was used

---

## References

1. Szegedy et al., 2014 — *Intriguing Properties of Neural Networks*
2. Goodfellow et al., 2015 — *Explaining and Harnessing Adversarial Examples*
3. Madry et al., 2017 — *[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)*
4. Dong et al., 2018 — *[Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081)*
5. Pomponi et al., 2022 — *[Pixle: a fast and effective black-box attack based on rearranging pixels](https://arxiv.org/abs/2202.02236)*
6. Kim, 2020 — *[Torchattacks: A PyTorch Repository for Adversarial Attacks](https://github.com/Harry24k/adversarial-attacks-pytorch)*

---

## Authors

- Michael Sensale
- Ali Alshehhi
- Varshan Muhunthan
- Poonyapat Sinpanyalert

---
