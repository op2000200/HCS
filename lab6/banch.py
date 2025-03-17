import torch
import time
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from tqdm.auto import tqdm

from lab5_load import MyNetDefault
from lab5_ver3 import MyNetCastom

def prepare_cifar10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(227, 227), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = datasets.CIFAR10

    training_data = ds(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    test_data = ds(
        root='data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        training_data, batch_size=batch_size,
        shuffle=True, num_workers=2,
        drop_last=True
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size,
        shuffle=False, num_workers=2,
        drop_last=True
    )

    return train_loader, test_loader

def benchmark_model(model_class, train_data, test_data, epochs=1, device='cuda:0'):
    model = model_class(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    start_time = time.time()
    for e in range(epochs):
        model.train()
        with tqdm(train_data, unit='batch') as epoch_loader:
            for inputs, labels in epoch_loader:
                x, y = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                y_ = model(x)
                loss = criterion(y_, y)
                loss.backward()
                optimizer.step()
    training_time = time.time() - start_time

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_data:
            x, y = inputs.to(device), labels.to(device)
            y_ = model(x)
    testing_time = time.time() - start_time

    return training_time, testing_time

if __name__ == '__main__':
    train_data, test_data = prepare_cifar10()

    print("Benchmarking MyNetDefault...")
    default_training_time, default_testing_time = benchmark_model(MyNetDefault, train_data, test_data, epochs=1)
    print(f"MyNetDefault - Training Time: {default_training_time:.2f}s, Testing Time: {default_testing_time:.2f}s")

    print("Benchmarking MyNetCastom...")
    castom_training_time, castom_testing_time = benchmark_model(MyNetCastom, train_data, test_data, epochs=1)
    print(f"MyNetCastom - Training Time: {castom_training_time:.2f}s, Testing Time: {castom_testing_time:.2f}s")
