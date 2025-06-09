import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)     # 3 channels, 64 filters, 3*3
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)    # 64 channels, 64 filters, 3*3
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)     # Batch-Norm
        x = F.gelu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)     # Batch-Norm
        x = F.gelu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

