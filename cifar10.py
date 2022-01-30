import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
from model import SimpleCNN
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
transform = transforms.Compose(
    [transforms.ToTensor(),
     ]
)

train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")
input = 32*32
output = 100



def get_accuracy(model, data_loader, device):
    accuracy = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        prediction = torch.argmax(output, dim=1)

        accuracy += (prediction == y).float().mean()

    return accuracy / len(data_loader)


# dims = [256 ,512, 1024, 2048]  # hyperparmeter1
dims = [128]
training_epochs = 500  # hyperparmter2
lr = 0.001
batch_size = 16

train_loader = DataLoader(train, shuffle=True, drop_last=True, batch_size=batch_size)
test_loader = DataLoader(test, shuffle=True, drop_last=True, batch_size=batch_size)


for dim in dims:
    model = SimpleCNN(hidden_dims=dim,output_dim=10).to(device)
    optimizers = [optim.Adam(model.parameters(), lr=lr), optim.SGD(model.parameters(),lr=lr), optim.Adadelta(model.parameters(),lr=lr),
                  optim.Adamax(model.parameters(),lr=lr),optim.AdamW(model.parameters(),lr=lr), optim.RMSprop(model.parameters(),lr=lr)]  # hyperparmeters3
    # result = {'norm1': [], 'norm2': [], 'norm3': [], 'mean1': [], 'mean2': [], 'mean3': [], 'var1': [], 'var2': [], 'var3': []}
    result = {'norm1': [], 'norm2': [], 'mean1': [], 'mean2': [], 'var1': [], 'var2': []}
    weight = {'layer1':[], 'layer2':[]}
    criterion = nn.CrossEntropyLoss()
    print(f"dim: {dim}  ")
    for opt in optimizers:
        optimizer = opt
        weight0 = []
        print(f"optimizer: {opt}")
        norm1, norm2 = model.weight_norm()
        mean1, mean2 = model.weight_mean()
        var1, var2 = model.weight_var()
        result['norm1'].append(norm1)
        result['norm2'].append(norm2)
        # result['norm3'].append(norm3)
        result['mean1'].append(mean1)
        result['mean2'].append(mean2)
        # result['mean3'].append(mean3)
        result['var1'].append(var1)
        result['var2'].append(var2)
        # result['var3'].append(var3)
        print(f"epoch: 0"
              f"\n[layer1] norm: {norm1:4f} mean:{mean1} var:{var1} "
              f"\n[layer2] norm: {norm2:4f} mean:{mean2} var:{var2} ")
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
                norm1, norm2, norm3, norm4 = model.weight_norm()
                mean1, mean2, mean3, mean4 = model.weight_mean()
                var1, var2, var3, var4 = model.weight_var()
                accuracy = get_accuracy(model, test_loader, device)
                result['norm1'].append(norm1)
                result['norm2'].append(norm2)
                # result['norm3'].append(norm3)
                result['mean1'].append(mean1)
                result['mean2'].append(mean2)
                # result['mean3'].append(mean3)
                result['var1'].append(var1)
                result['var2'].append(var2)
                # result['var3'].append(var3)
                print(f"epoch: 0"
                      f"\n[layer1] norm: {norm1:4f} mean:{mean1} var:{var1} "
                      f"\n[layer2] norm: {norm2:4f} mean:{mean2} var:{var2} "
                      f"\nloss accuracy: {accuracy:4f}")


pd.DataFrame(result).to_csv('result_cifar10.csv', index=False) # 저장-형식 고민




