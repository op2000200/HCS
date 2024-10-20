import torch
import lab2





def test1():
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
    res = [1,2,3,4]
    print(res)
    assert 1 == 1

