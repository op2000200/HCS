import torch
import lab2


# Assertion for test case 1
def test_1_assertion(res):
    return (res[0] == 3) and (res[1] == 11) and (res[2] == 23) and (res[3] == 39) and (res[4] == 59)


# Test case 1
# CPU
def test_cpu_1():
    vec1 = torch.empty(5)
    vec2 = torch.empty(5)
    vec3 = torch.empty(5)
    vec4 = torch.empty(5)

    for i in range(5):
        vec1[i] = i
        vec2[i] = i + 1
        vec3[i] = i + 2
        vec4[i] = i + 3

    res = lab2.cpu(vec1,vec2,vec3,vec4)

    assert test_1_assertion(res)

# Test case 1
# GPU
def test_gpu_1():
    vec1 = torch.empty(5)
    vec2 = torch.empty(5)
    vec3 = torch.empty(5)
    vec4 = torch.empty(5)

    for i in range(5):
        vec1[i] = i
        vec2[i] = i + 1
        vec3[i] = i + 2
        vec4[i] = i + 3

    res = lab2.gpu(vec1,vec2,vec3,vec4)

    assert test_1_assertion(res)

