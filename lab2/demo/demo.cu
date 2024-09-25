#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "lib/scmp.cu"

__managed__ float* arr1, float* arr2, float* arr3;

int main()
{
    arr1[0] = 0.f;
    arr2[0] = 0.f;
    scmpOnCPU(arr1, arr2, arr3, 0);
    std::cout << arr3[0];
}