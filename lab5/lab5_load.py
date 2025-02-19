import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from tqdm.auto import tqdm
#from moduleNew import LinearFunction, LinearLayer

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

class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()

        self.pipe = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, x):
        return self.pipe(x)

def test(net, test_data, num_classes,
         device='cuda:0', bar_label='', use_saved_model=True):

    t = torch.zeros(num_classes, device=device)
    f = torch.zeros(num_classes, device=device)

    if use_saved_model:
        net.load_state_dict(torch.load('model_acc_0.40915459394454956.pt'))

    net.to(device)

    training_mode = net.training
    if training_mode:
        net.eval()

    if not bar_label:
        bar_label = 'Testing'

    with tqdm(test_data, unit='batch', desc=bar_label) as test:
        for inputs, labels in test:
            x, y = inputs.to(device), labels.to(device)

            _, predicted = torch.max(net(x).data, 1)

            t[y] += (predicted == y)
            f[y] += (predicted != y)

    acc = t / (t + f) * 100

    if training_mode:
        net.train()

    return acc


if __name__ == '__main__':
    train_data, test_data = prepare_cifar10()
    net = MyNet(num_classes=10)

    valid_accuracy_cls = test(net, test_data, num_classes=10)

    print('\nAccuracy (valid)')
    print(f'  min: {valid_accuracy_cls.min().item():.0f}%')
    print(f'  avg: {valid_accuracy_cls.mean().item():.0f}%')
    print(f'  max: {valid_accuracy_cls.max().item():.0f}%\n')