#include <torch/extension.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void scmpOnGPU(float* vector_a, float* vector_b, float* result)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = vector_a[i] * vector_b[i];
}

__host__ void scmpOnCPU(const float* vector_a, const float* vector_b, int size, float* result, int n)
{
    *result = 0;
    for (int i = 0; i < n; i++)
    {
        *result += vector_a[i] +vector_b[i];
    }
}

float calcOnGpu(torch::Tensor vec1, torch::Tensor vec2)
{
    int size = vec1.size(0);
    float *d_vector_a, *d_vector_b, *d_result;
    float *res = new float[size];
    cudaMalloc(&d_vector_a, sizeof(float) * size);
    cudaMalloc(&d_vector_b, sizeof(float) * size);
    cudaMalloc(&d_result, sizeof(float) * size);

    cudaMemcpy(d_vector_a, vec1.data_ptr<float>(), sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, vec2.data_ptr<float>(), sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, res, sizeof(float) * size, cudaMemcpyHostToDevice);

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
    
    scmpOnGPU <<<bl, th >>> (d_vector_a, d_vector_b, d_result);

    cudaDeviceSynchronize();

    // Копируем результат обратно на хост
    cudaMemcpy(res, d_result, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // Освобождаем память на устройстве
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_result);
    float sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        sum += res[i];
    }
    
    return sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_gpu", &calcOnGpu);
}