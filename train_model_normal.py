import argparse
from tqdm import tqdm
from load_MNIST_data import getMNISTDataLoaders
import torch.nn as nn
import torchvision.models as models
import torch
import os
from torchattack import PGD

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--robust', action='store_true')
parser.add_argument('--attack', type=str, default='pgd', choices=['pgd'])

args = parser.parse_args()

train_loader, val_loader, test_loader = getMNISTDataLoaders(batchSize=args.batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'resnet18':
    model = models.resnet18(num_classes=10, weights=None)
elif args.model == 'resnet50':
    model = models.resnet50(num_classes=10, weights=None)

model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model = model.to(device)

xent_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

model.train()
n_epochs = args.epochs
lr = args.lr

for i in range(n_epochs):
    for j, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Batch'):
        images, labels = images.to(device), labels.to(device)

        if args.robust:
            model.eval()
            attack = PGD(model, eps=0.3, alpha=2/255, steps=7, random_start=True)
            images = attack(images, labels)
            model.train()

        optimizer.zero_grad()
        outputs = model(images)
        loss = xent_loss(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

os.makedirs('models', exist_ok=True)
if args.robust:
    torch.save(model.state_dict(), f'models/{args.model}_mnist_robust_{args.attack}.pth')
else:
    torch.save(model.state_dict(), f'models/{args.model}_mnist_normal.pth')

print(f'Accuracy of the model on the test images: {100 * correct / total} %')