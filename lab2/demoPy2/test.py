import torch
import lab2

vec1 = torch.empty(5)
vec2 = torch.empty(5)
vec3 = torch.empty(5)
vec4 = torch.empty(5)

for i in range(5):
    vec1[i] = i
    vec2[i] = i + 1
    vec3[i] = i + 2
    vec4[i] = i + 3

print(vec1)
print(vec2)
print(vec3)
print(vec4)
res1 = lab2.cpu(vec1,vec2,vec3,vec4)
print(res1)
res2 = lab2.gpu(vec1,vec2,vec3,vec4)
print(res2)