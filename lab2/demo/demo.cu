#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "lib/scmp.cu"

const int n = 5;
const size_t size = sizeof(float) * n;
//for host
float *vector_a_x, *vector_a_y, *vector_b_x, *vector_b_y, *result;
//for device
float *d_vector_a_x, *d_vector_a_y, *d_vector_b_x, *d_vector_b_y, *d_result;

void allocateMemory();
void loadValues();
void calcOnCPU();
void calcOnGPU();
void clear();

int main()
{
    allocateMemory();
    loadValues();
    calcOnCPU();
    calcOnGPU();
    clear();
}

void allocateMemory()
{
    vector_a_x = (float*)malloc(size);
    vector_a_y = (float*)malloc(size);
    vector_b_x = (float*)malloc(size);
    vector_b_y = (float*)malloc(size);
    result = (float*)malloc(size);
    cudaMalloc(&d_vector_a_x,size);
    cudaMalloc(&d_vector_a_y,size);
    cudaMalloc(&d_vector_b_x,size);
    cudaMalloc(&d_vector_b_y,size);
    cudaMalloc(&d_result,size);
}

void loadValues()
{
    srand(0);
    for (size_t i = 0; i < n; i++)
    {
        vector_a_x[i] = (float(rand()+1) / float(rand()+1));
        vector_a_y[i] = (float(rand()+1) / float(rand()+1));
        vector_b_x[i] = (float(rand()+1) / float(rand()+1));
        vector_b_y[i] = (float(rand()+1) / float(rand()+1));
        std::cout << vector_a_x[i] << " " << vector_a_y[i] << " " << vector_b_x[i] << " " << vector_b_y[i] << std::endl;
    }
    std::cout << std::endl;
}

void calcOnCPU()
{
    scmpOnCPU(vector_a_x, vector_a_y, vector_b_x, vector_b_y, result, n);

    for (size_t i = 0; i < n; i++)
    {
        std::cout << result[i] << std::endl;
    }
    std::cout << std::endl;
}

void calcOnGPU()
{
    cudaMemcpy(d_vector_a_x, vector_a_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_a_y, vector_a_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b_x, vector_b_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b_y, vector_b_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, size, cudaMemcpyHostToDevice);

    int bl, th;
    if (n > 1024)
    {
        bl = (n / 1024) + 1;
        th = 1024;
    }
    else
    {
        th = n;
        bl = 1;
    }

    scmpOnGPU <<<bl, th >>> (d_vector_a_x, d_vector_a_y, d_vector_b_x, d_vector_b_y, d_result);

    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; i++)
    {
        std::cout << result[i] << std::endl;
    }
    std::cout << std::endl;
}

void clear()
{
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
}