import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import pandas as pd
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x.view(-1))
     ]
)

train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(28*28, self.dim)
        self.linear2 = nn.Linear(self.dim, 10)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)

        return out

    def fc_weight(self):
        weight = self.linear1.weight.cpu().detach().numpy()
        return weight

    def norm(self):
        weight1 = self.linear1.weight.cpu().detach().numpy()
        weight2 = self.linear2.weight.cpu().detach().numpy()

        norm1 = np.mean(np.linalg.norm(weight1, axis=1))
        norm2 = np.mean(np.linalg.norm(weight2, axis=1))

        return norm1, norm2


def get_accuracy(model, data_loader, device):
    accuracy = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        prediction = torch.argmax(output, dim=1)

        accuracy += (prediction == y).float().mean()

    return accuracy / len(data_loader)


dims = [256]  # hyperparmeter1
training_epochs = 10000  # hyperparmter2
lr = 0.001
#optimizer = optim.Adam #  hyperparmeters3
batch_size = 64

train_loader = DataLoader(train, shuffle=True, drop_last=True, batch_size=batch_size)
test_loader = DataLoader(test, shuffle=True, drop_last=True, batch_size=batch_size)
result = []


for dim in dims:
    model = Model(dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(training_epochs)):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        if (epoch + 1) % 10 == 0:
            norm1, norm2 = model.norm()
            accuracy = get_accuracy(model, test_loader, device)
            result.append([epoch, accuracy ,norm1, norm2])
            print(f"epooch: {epoch +1} loss:{loss} dim: {dim} norm1:{norm1:.4f} norm2:{norm2:.4f} accuracy: {accuracy:.4f}")

pd.DataFrame(result).to_csv('result_mnist_10000.csv', index=False) # 저장-형식 고민