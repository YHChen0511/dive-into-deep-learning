from mlp import MLP
from cnn import CovLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import tqdm
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])




class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1000, shuffle=False)
    
    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for image, label in tqdm.tqdm(train_loader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            predict = model(image)
            loss = F.cross_entropy(predict, label)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                correct += (y_hat.argmax(1) == y).sum().item()
        print(f"Epoch: {epoch}, Accuracy: {correct/len(test_dataset)}")


