import torch.nn as nn
import torch
import torch.nn.functional as F
import tqdm
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

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
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=args.data, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root=args.data, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1000, shuffle=False)

    if not os.path.exists('mnist_cnn.pth'):
        print("Model not found. Training...")
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

        torch.save(model.state_dict(), 'mnist_cnn.pth')
        print("Model saved.")
    
    else :
        model = Model().to(device)
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        print("Model loaded")

    # Save example images
    
    i = 0
    for test_image, gt_label in test_dataset:
        if(i >= 4):
            break
        test_image = test_image.to(device)
        predict = model(test_image)
        predict_label = predict.argmax(1).item()
        plt.imshow(test_image.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Predict: {predict_label}, GT: {gt_label}")
        plt.savefig(f'{args.output}/example_{i}.png')
        i += 1
    print("Example images saved.")

        
        


    


