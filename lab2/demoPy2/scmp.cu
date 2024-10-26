#include <torch/extension.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void scmpOnGPU(float* vector_a, float* vector_b, int size, float* result)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    *result = vector_a[i] * vector_b[i];
}

__host__ void scmpOnCPU(const float** vector_a, const float** vector_b, int size, float* result, int n)
{
    for (int i = 0; i < n; i++)
    {
        float buffer = 0;
        for (int j = 0; j < size; j++)
        {
            buffer += vector_a[i][j] * vector_b[i][j];
        }
        result[i] = buffer;
    }
}

float calcOnGpu(torch::Tensor vec1, torch::Tensor vec2)
{
    int size = vec1.size(0);
    float *d_vector_a, *d_vector_b, *d_result;
    float buf = 0.f;
    cudaMalloc(&d_vector_a, sizeof(float) * size);
    cudaMalloc(&d_vector_b, sizeof(float) * size);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_vector_a, vec1.data_ptr<float>(), sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, vec1.data_ptr<float>(), sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &buf, sizeof(float), cudaMemcpyHostToDevice);

    *d_result = 0;

    int bl, th;
    if (size > 1024)
    {
        bl = (size / 1024) + 1;
        th = 1024;
    }
    else
    {
        th = size;
        bl = 1;
    }
    
    scmpOnGPU <<<bl, th >>> (d_vector_a, d_vector_b, size, d_result);

    cudaDeviceSynchronize();
    
    float res = *d_result;
    std::cout << res << std::endl;
    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_gpu", &calcOnGpu);
}