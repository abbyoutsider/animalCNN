import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 128 * 128, 5)  # Update if input size changes
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AnimalSpeciesCNN(nn.Module):
    def __init__(self):
        super(AnimalSpeciesCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # Reduce filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample (256 -> 128)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # Reduce filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample (128 -> 64)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Reduce filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample (64 -> 32)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten into (batch, 32*32*64)
            nn.Linear(32*32*32, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 5)  # Output layer for 5 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x