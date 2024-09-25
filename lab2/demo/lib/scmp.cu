#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void scmpOnGPU(const float* num1, const float* num2, float* result)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = 1.f;
}

__host__ void scmpOnCPU(const float* num1, const float* num2, float* result, const int i)
{
    result[i] = 2.f;
}