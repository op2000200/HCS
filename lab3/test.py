import torch
import lab2

vec1 = torch.empty(5)
vec2 = torch.empty(5)

for i in range(5):
    vec1[i] = i
    vec2[i] = i + 1

print(vec1)
print(vec2)
res = lab2.dot_gpu(vec1,vec2)
print(res)