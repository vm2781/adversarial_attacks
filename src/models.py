import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet1_0

class LeNet5(nn.Module):
    """
    LeNet-5 architecture for MNIST classification.
    Classic CNN designed for handwritten digit recognition.

    Architecture:
    - Input: 28x28 grayscale images
    - Conv1: 6 filters, 5x5 kernel
    - MaxPool: 2x2
    - Conv2: 16 filters, 5x5 kernel
    - MaxPool: 2x2
    - FC1: 120 units
    - FC2: 84 units
    - Output: 10 classes

    Total parameters: ~61k
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv1: 28x28 -> 24x24
        x = F.relu(self.conv1(x))
        # MaxPool: 24x24 -> 12x12
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Conv2: 12x12 -> 8x8
        x = F.relu(self.conv2(x))
        # MaxPool: 8x8 -> 4x4
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 16*4*4)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class SqueezeNetMNIST(nn.Module):
    """SqueezeNet adapted for MNIST by converting grayscale to RGB."""

    def __init__(self):
        super(SqueezeNetMNIST, self).__init__()
        base_model = squeezenet1_0(pretrained=False)
        # Replace final classifier to output 10 classes instead of 1000
        base_model.classifier[1] = nn.Conv2d(512, 10, kernel_size=1)
        self.model = base_model

    def forward(self, x):
        # Expand 1-channel to 3-channel by repeating
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)
