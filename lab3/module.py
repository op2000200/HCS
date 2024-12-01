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
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = self.weight.cuda()
        self.bias = torch.empty(out_features, **factory_kwargs)
        self.bias = self.bias.cuda()
        self.reset_parametrs()

    def forward(self, x):
        return lab3.linear_layer_calc_result(x, self.weight, self.bias)
    
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