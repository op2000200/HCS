
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

__global__ void sum(const float* a, const float* b, float* c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    c[j] = a[j] + b[j];
}

__host__ float sumH(const float a, const float b)
{
    float res = a + b;
    return res;
}

const int n = 10000000;

__managed__ float vector_a[n], float vector_b[n], float vector_c[n], float vector_d[n];

int main()
{
    std::ifstream in;
    std::ifstream in2;
    std::ofstream out1;
    std::ofstream out2;
    in.open("val3.txt");
    in2.open("val4.txt");
    out1.open("resgpu.txt");
    out2.open("rescpu.txt");

    int valSize;
    std::string buffer;
    in >> buffer;
    in2 >> buffer;
    valSize = std::stoi(buffer);

    if (valSize >= n)
    {
        std::cout << "working" << std::endl;
        for (int i = 0; i < n; i++)
        {
            in >> buffer;
            vector_a[i] = std::stof(buffer);
            in2 >> buffer;
            vector_b[i] = std::stof(buffer);
        }

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

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        sum << <bl, th >> > (vector_a, vector_b, vector_c);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsedTimeGPU;
        cudaEventElapsedTime(&elapsedTimeGPU, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        cudaDeviceSynchronize();
        elapsedTimeGPU = elapsedTimeGPU;
        std::cout << "Count: " << n << std::endl << "elapsed time GPU: " << elapsedTimeGPU << " ms" << std::endl;

        //for (int i = 0; i < n; i++)
        //{
        //    out1 << vector_c[i] << std::endl;
        //}
        std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++)
        {
            vector_d[i] = sumH(vector_a[i], vector_b[i]);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration dur = end - st;
        float elapsedTimeCPU = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        std::cout << "elapsed time CPU: " << elapsedTimeCPU <<" ms" << std::endl;
        std::cout << "acceleration: " << ((elapsedTimeCPU > elapsedTimeGPU) ? elapsedTimeCPU / elapsedTimeGPU : elapsedTimeGPU / elapsedTimeCPU) << std::endl;
        //for (int i = 0; i < n; i++)
        //{
        //    out2 << vector_d[i] << std::endl;
        //}
        std::cout << "done" << std::endl;
    }
    else
    {
        std::cout << "error" << std::endl;
    }
    in.close();
    in2.close();
    out1.close();
    out2.close();
}
