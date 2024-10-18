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

int check(int in)
{
    return in;
}