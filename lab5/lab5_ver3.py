import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from tqdm.auto import tqdm
from moduleNew import LinearFunction, LinearLayer


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
            LinearLayer(in_features=9216, out_features=4096), nn.ReLU(), nn.Dropout(p=0.5),
            LinearLayer(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout(p=0.5),
            #nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            #nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, x):
        return self.pipe(x)


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


def test_accuracy(y_, y, num_batchs=1):
    with torch.no_grad():
        _, predicted = torch.max(y_, 1)
        accuracy = (predicted == y).sum() / (num_batchs * y.numel())
    return accuracy


def eval_valid_set_accuracy(net, test_data, device):
    accuracy = torch.zeros((1,), device=device)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            x, y = inputs.to(device), labels.to(device)
            y_ = net(x)
            accuracy += (
                test_accuracy(y_, y, len(test_data))
            )

    return accuracy


def check_loss_value(loss):
    with torch.no_grad():
        if loss.isinf() or loss.isnan():
            raise ValueError('Invalid loss.')

def improved_train_iteration(
    train_data, test_data,
    net, epochs,
    lr=0.005, device='cuda:0',
    bar_label='Training',
    patience=10,
    reduce_lr_factor=0.5,
    reduce_lr_patience=1,
    accuracy_history=False,
    make_checkpoints=True
):
    ret = {}
    best_valid_accuracy = 0.0
    epochs_no_improvement = 0

    train_accuracy = torch.zeros((epochs,), device=device) if accuracy_history else None
    valid_accuracy = torch.zeros((epochs,), device=device)

    if accuracy_history:
        ret['train_accuracy'] = train_accuracy
        ret['valid_accuracy'] = valid_accuracy

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=reduce_lr_factor, patience=reduce_lr_patience)

    for e in range(0, epochs):
        net.train()
        epoch_loss = 0.0

        with tqdm(train_data, unit='batch') as epoch_loader:
            epoch_loader.set_description(f'{bar_label} Epoch {e + 1}/{epochs}')

            for i, (inputs, labels) in enumerate(epoch_loader):
                x, y = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                y_ = net(x)
                loss = criterion(y_, y)
                loss.backward() 
                optimizer.step()

                epoch_loss += loss.item() 

                if accuracy_history:
                    train_accuracy[e] += test_accuracy(y_, y, len(train_data))

                epoch_loader.set_postfix(loss=loss.item())

        net.eval()
        valid_accuracy[e] = eval_valid_set_accuracy(net, test_data, device).item()

        if valid_accuracy[e] > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy[e]
            epochs_no_improvement = 0
            if make_checkpoints:
                torch.save(net.state_dict(), f'model_acc_{best_valid_accuracy}.pt')
        else:
            epochs_no_improvement += 1

        scheduler.step(valid_accuracy[e])

        if epochs_no_improvement >= patience:
            print("Early stopping...")
            break

    ret['best_valid_accuracy'] = best_valid_accuracy
    return ret



def test(net, test_data, num_classes,
         device='cuda:0', bar_label='', use_saved_model=False):

    t = torch.zeros(num_classes, device=device)
    f = torch.zeros(num_classes, device=device)

    if use_saved_model:
        torch.save(net.state_dict(), 'model.pt')
        net.load_state_dict(torch.load('best_model.pt')) #WRITE NEW PATH

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

    if use_saved_model:
        net.load_state_dict(torch.load('model.pt'))

    return acc


if __name__ == '__main__':
    train_data, test_data = prepare_cifar10()

    #LinearFunction.up_backend('hs/lab3/lab3g3d.cu')
    net = MyNet(num_classes=10)

    improved_train_iteration(train_data, test_data, net, epochs=100, lr=0.005)

    valid_accuracy_cls = test(net, test_data, num_classes=10)

    print('\nAccuracy (valid)')
    print(f'  min: {valid_accuracy_cls.min().item():.0f}%')
    print(f'  avg: {valid_accuracy_cls.mean().item():.0f}%')
    print(f'  max: {valid_accuracy_cls.max().item():.0f}%\n')