import torch
import torch.nn as nn
import math
import lab3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w, b)

        y = lab3.linear_layer_calc_result(x, w, b)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, w, b = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = lab3.linear_layer_calc_grads(x, w, grad_output)
        return grad_input, grad_weight, grad_bias
    
class SimpleNN(nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features, out_features, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs).cuda())
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs).cuda())
        self.reset_parametrs()

    def forward(self, x):
        self.y = Layer.apply(x, self.weight, self.bias)
        return self.y
    
    def reset_parametrs(self):
        nn.init.kaiming_uniform_ (self.weight, a = math.sqrt (5) )
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out ( self.weight )
        bound = 1 / math.sqrt ( fan_in ) if fan_in > 0 else 0
        nn.init.uniform_ (self.bias, - bound, bound )

X = torch.randn(16, 16, requires_grad=True, device=device)

# Создаем экземпляр модели
model = SimpleNN(16, 16).to(device)

# Получаем предсказание
output = model(X)
print("Output:", output)
loss = output.cuda().sum()
loss.backward()
print("Gradients:", X.grad)