import torch
import lab3

m = 3
k = 2
n = 1

X = torch.randn(m, k).cuda()
W = torch.randn(n, k).cuda()
b = torch.randn(n).cuda()

print(X)
print(W)
print(b)
print(lab3.linear_layer_forward(X, W, b))

linear_layer = torch.nn.Linear(k, n).cuda()
with torch.no_grad():
    linear_layer.weight.copy_(W)
    linear_layer.bias.copy_(b)
    print(linear_layer(X))