#include <torch/extension.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void scmpOnGPU(const float* vector_a_x, const float* vector_a_y, const float* vector_b_x, const float* vector_b_y, float* result)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = vector_a_x[i] * vector_b_x[i] + vector_a_y[i] * vector_b_y[i];
}

__host__ void scmpOnCPU(const float* vector_a_x, const float* vector_a_y, const float* vector_b_x, const float* vector_b_y, float* result, const int n)
{
    for (size_t i = 0; i < n; i++)
    {
        result[i] = vector_a_x[i] * vector_b_x[i] + vector_a_y[i] * vector_b_y[i];
    }
}


torch::Tensor test_dummy(torch::Tensor vec1, torch::Tensor vec2, torch::Tensor vec3, torch::Tensor vec4)
{
    torch::Tensor res = torch::empty(1);
    return res;
}


torch::Tensor calcOnCpu(torch::Tensor vec1, torch::Tensor vec2, torch::Tensor vec3, torch::Tensor vec4)
{
    int size = vec1.size(0);
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
    scmpOnCPU(vector_a_x, vector_a_y, vector_b_x, vector_b_y, result, size);
    
    torch::Tensor res = torch::empty(size);
    for (int i = 0; i < size; i++)
    {
      res[i] = result[i];
      std::cout << result[i] << " " << res[i] << std::endl;
    }

    delete[] vector_a_x;
    delete[] vector_a_y;
    delete[] vector_b_x;
    delete[] vector_b_y;
    delete[] result;

    return res;
}

torch::Tensor calcOnGpu(torch::Tensor vec1, torch::Tensor vec2, torch::Tensor vec3, torch::Tensor vec4)
{
    int size = vec1.size(0);
    float *vector_a_x, *vector_a_y, *vector_b_x, *vector_b_y, *result;
    float *d_vector_a_x, *d_vector_a_y, *d_vector_b_x, *d_vector_b_y, *d_result;
    vector_a_x = (float*)malloc(sizeof(float) * size);
    vector_a_x = vec1.data<float>();
    vector_a_y = (float*)malloc(sizeof(float) * size);
    vector_a_y = vec2.data<float>();
    vector_b_x = (float*)malloc(sizeof(float) * size);
    vector_b_x = vec3.data<float>();
    vector_b_y = (float*)malloc(sizeof(float) * size);
    vector_b_y = vec4.data<float>();
    result = (float*)malloc(sizeof(float) * size);
    cudaMalloc(&d_vector_a_x,sizeof(float) * size);
    cudaMalloc(&d_vector_a_y,sizeof(float) * size);
    cudaMalloc(&d_vector_b_x,sizeof(float) * size);
    cudaMalloc(&d_vector_b_y,sizeof(float) * size);
    cudaMalloc(&d_result,sizeof(float) * size);
    cudaMemcpy(d_vector_a_x, vector_a_x, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_a_y, vector_a_y, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b_x, vector_b_x, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b_y, vector_b_y, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, sizeof(float) * size, cudaMemcpyHostToDevice);
    
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
    
    scmpOnGPU <<<bl, th >>> (d_vector_a_x, d_vector_a_y, d_vector_b_x, d_vector_b_y, d_result);

    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, sizeof(float) * size, cudaMemcpyDeviceToHost);
    
    torch::Tensor res = torch::empty(size);
    for (int i = 0; i < size; i++)
    {
      res[i] = result[i];
      std::cout << result[i] << " " << res[i] << std::endl;
    }

    cudaFree(d_vector_a_x);
    cudaFree(d_vector_a_y);
    cudaFree(d_vector_b_x);
    cudaFree(d_vector_b_y);
    cudaFree(d_result);
    delete[] vector_a_x;
    delete[] vector_a_y;
    delete[] vector_b_x;
    delete[] vector_b_y;
    delete[] result;

    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu", &calcOnCpu);
    m.def("gpu", &calcOnGpu);
    m.def("test", &test_dummy);
