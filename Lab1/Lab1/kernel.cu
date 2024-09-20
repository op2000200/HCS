
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>

__global__ void add(int* a, int* b)
{
    int i = threadIdx.x;
    //c[i] = a[i] + b[i];
}

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

const int n = 4000;

__managed__ float vector_a[n];
__managed__ float vector_b[n];
__managed__ float vector_c[n];
__managed__ float vector_d[n];

int main()
{
    std::ifstream in;
    std::ifstream in2;
    std::ofstream out1;
    std::ofstream out2;
    in.open("val1.txt");
    in2.open("val2.txt");
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
        std::cout << "Will execute " << n << " times...\n";
        for (int i = 0; i < n; i++)
        {
            std::cout << "LUL " << i << std::endl;

            in >> buffer;
            //vector_a[i] = std::stof(buffer);
            vector_a[i] = i;
            in2 >> buffer;
            //vector_b[i] = std::stof(buffer);
            vector_b[i] = i;
        }

        std::cout << "TEST TEST TEST" << std::endl;

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

        //std::cout << bl << " " << th << std::endl << std::endl;

        
        sum <<<bl, th >>> (vector_a, vector_b, vector_c);
        

        cudaDeviceSynchronize();

        for (int i = 0; i < n; i++)
        {
            out1 << vector_c[i] << std::endl;
        }

        for (int i = 0; i < n; i++)
        {
            vector_d[i] = sumH(vector_a[i], vector_b[i]);
        }

        for (int i = 0; i < n; i++)
        {
            out2 << vector_d[i] << std::endl;
        }
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
