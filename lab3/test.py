import torch
import lab3

m = 3  # количество примеров
k = 2  # количество входных признаков
n = 1  # количество выходных признаков

# Создайте тензоры
X = torch.randn(m, k).cuda()  # Входной тензор
W = torch.randn(n, k).cuda()  # Тензор весов
b = torch.randn(n).cuda()      # Тензор смещений

# Вычисление с помощью расширения
print(X)
print(W)
print(b)
print(lab3.linear_layer_forward(X, W, b))


linear_layer = torch.nn.Linear(k, n).cuda()
with torch.no_grad():
    linear_layer.weight.copy_(W)  # Устанавливаем веса
    linear_layer.bias.copy_(b)     # Устанавливаем смещения
    print(linear_layer(X))     # Вызов библиотечной функции