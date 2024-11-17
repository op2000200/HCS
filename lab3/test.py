import torch
import lab3

m = 2
k = 2
n = 1

X = torch.randn(m, k).cuda()
W = torch.randn(n, k).cuda()
b = torch.randn(n).cuda()

print("X:\n", X,"\n-------\n")
print("W:\n",W,"\n-------\n")
print("b:\n",b,"\n-------\n")
res = lab3.linear_layer_calc_result(X, W, b)
print("Y(X) our:\n",res,"\n-------\n")

linear_layer = torch.nn.Linear(k, n).cuda()
with torch.no_grad():
    linear_layer.weight.copy_(W)
    linear_layer.bias.copy_(b)
    print("Y(X) pytorch:\n",linear_layer(X),"\n-------\n")
print("Grad of X:\n",lab3.linear_layer_calc_grads(X, W, res)[0],"\n Grad of W\n-------\n",lab3.linear_layer_calc_grads(X, W, res)[1],"\n Grad of b\n-------\n",lab3.linear_layer_calc_grads(X, W, res)[2])

