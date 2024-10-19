#include <torch/extension.h>
#include "scmp.cu"

int checkk(int in)
{
    return in;
}

torch::Tensor calcOnCpu(torch::Tensor vec1, torch::Tensor vec2, torch::Tensor vec3, torch::Tensor vec4, int size)
{
    float *vector_a_x, *vector_a_y, *vector_b_x, *vector_b_y, *result;
    vector_a_x = (float*)malloc(sizeof(float) * size);
    vector_a_x = vec1.data<float>();
    vector_a_y = (float*)malloc(sizeof(float) * size);
    vector_a_y = vec2.data<float>();
    vector_b_x = (float*)malloc(sizeof(float) * size);
    vector_b_x = vec3.data<float>();
    vector_b_y = (float*)malloc(sizeof(float) * size);
    vector_b_y = vec4.data<float>();
    result = (float*)malloc(sizeof(float) * size);
    scmpOnCPU(vector_a_x, vector_a_y, vector_b_x, vector_b_y, result, n);
    
    torch::Tensor res = torch::empty(size);
    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check", &checkk);
    m.def("add", &calcOnCpu);
}