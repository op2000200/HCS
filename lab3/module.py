import torch
import torch.nn as nn
import math
import lab3

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
        self.x = x
        self.y = lab3.linear_layer_calc_result(x, self.weight, self.bias)
        return self.y

    def backward(self):
        return lab3.linear_layer_calc_grads(self.x, self.weight, self.y)
    
    def reset_parametrs(self):
        nn . init . kaiming_uniform_ (self.weight , a = math . sqrt (5) )
        fan_in , _ = nn . init . _calculate_fan_in_and_fan_out ( self.weight )
        bound = 1 / math . sqrt ( fan_in ) if fan_in > 0 else 0
        nn . init . uniform_ (self.bias , - bound , bound )

X = torch.randn(5, 4).cuda()

# Создаем экземпляр модели
model = SimpleNN(5, 3)

# Получаем предсказание
output = model(X)
print("Output:", output)
output = model.backward()
print("Output:", output)