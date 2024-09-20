
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>


__global__ void sum(const float* a, const float* b, float* c);

__host__ float sumH(const float a, const float b);
