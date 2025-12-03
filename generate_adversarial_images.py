import torch
from torchattack import PGD
from torchattacks.attacks.cw import CW
from load_MNIST_data import getMNISTDataLoaders
import torchvision.models as models
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--attack', type=str, default='pgd', choices=['pgd', 'cw'])
parser.add_argument('--model_path', type=str, default='models/resnet18_mnist_normal.pth')
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)


train_loader, val_loader, test_loader = getMNISTDataLoaders(batchSize=args.batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == 'resnet18':
    model = models.resnet18(num_classes=10, weights=None)
elif args.model == 'resnet50':
    model = models.resnet50(num_classes=10, weights=None)

model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model = model.to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

print(f"Using device: {device}")

examples_found = 0
total_examples = 0

adversarial_dataset = []
fooled_dataset = []

true_labels = []
true_labels_fooled = []

for batch_idx, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Generating Adversarial Examples'):
    # if examples_found >= target_count:
        # break

    images, labels = images.to(device), labels.to(device)

    # Generate adversarial examples
    if args.attack == 'cw':
        adversary = CW(model, c=1, kappa=0, steps=50, lr=0.01)
    elif args.attack == 'pgd':
        adversary = PGD(model, eps=0.3, steps=7, random_start=True)
    adv_images = adversary(images, labels)

    # Get model predictions on adversarial examples
    with torch.no_grad():
        logits = model(adv_images)
        _, preds = torch.max(logits, 1)

    # Find which ones actually fool the model (prediction != true label)
    fooled_mask = (preds != labels)
    examples_found += fooled_mask.sum().item()
    total_examples += images.shape[0]

    adversarial_dataset.append(adv_images.cpu())
    fooled_dataset.append(adv_images[fooled_mask].cpu())
    true_labels.append(labels.cpu())
    true_labels_fooled.append(labels[fooled_mask].cpu())


torch.save({
    'image': torch.cat(adversarial_dataset),
    'label': torch.cat(true_labels)
}, f'{args.output_dir}/all_adversarial_images_{args.attack}.pth')

torch.save({
    'image': torch.cat(fooled_dataset),
    'label': torch.cat(true_labels_fooled)
}, f'{args.output_dir}/fooled_images_{args.attack}.pth')

print(f"\nTotal fooling examples found: {examples_found}/{total_examples}")
print(f"All files saved to : {args.output_dir}")


print(f"Accuracy on Adversarial Images: {100 * (1- examples_found / total_examples)}")