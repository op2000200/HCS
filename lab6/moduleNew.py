import torch
import torch.nn as nn
import math
import lab3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w, b)
        return lab3.linear_layer_calc_result(x, w, b)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, _ = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = lab3.linear_layer_calc_grads(x, w, grad_output)
        
        return grad_input, grad_weight, grad_bias

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        ops = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **ops),
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, **ops),
        )

        self.reset_parametrs()

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)

    def reset_parametrs(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


if __name__ == "__main__":
    model = LinearLayer(100, 100).to(DEVICE)
    x = torch.randn(100, 100, device=DEVICE)
    y = model(x)
    y.backward(torch.ones_like(y))

    print("Output:", y)
    print("Gradients:")
    for p in model.parameters():
        print(p.grad)
