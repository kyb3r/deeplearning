import torch 
from torch import nn 
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, identity_downsample=None):
        super().__init__()
        self.expansion = 4
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        
        
    def forward(self, x):
        identity = x
        x = self.layers(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module): # [3, 4, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet_layers = nn.Sequential(
            self._make_layer(block, layers[0], out_channels=64, stride=1),
            self._make_layer(block, layers[1], out_channels=128, stride=2),
            self._make_layer(block, layers[2], out_channels=256, stride=2),
            self._make_layer(block, layers[3], out_channels=512, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet_layers(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    @classmethod
    def ResNet50(cls, image_channels=3, num_classes=1000):
        return cls(Block, [3, 4, 6, 3], image_channels, num_classes)

    @classmethod
    def MNIST(cls, image_channels=1, num_classes=10):
        return cls(Block, [1, 1, 1, 1], image_channels, num_classes)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4),
            )
        
        layers.append(block(self.in_channels, out_channels, stride, identity_downsample))
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)



# Train mnist with resnet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        running_corrects += torch.sum(preds == target.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, target)

            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == target.data)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    print(f"Test Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet.MNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    print(f"Number of parameters: {model.num_params}")
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/{10}")
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion, epoch)
        print()
    
if __name__ == "__main__":
    main()
