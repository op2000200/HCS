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

at::Tensor calcOnGpu(at::Tensor &vec1, at::Tensor &vec2, at::Tensor &vec3, at::Tensor &vec4)
{
    //torch::Device device(torch::kCUDA)
    int size = vec1.size(0);
    at::Tensor vec1_contig = vec1.contiguous();
    at::Tensor vec2_contig = vec2.contiguous();
    at::Tensor vec3_contig = vec3.contiguous();
    at::Tensor vec4_contig = vec4.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float* vec1_contig_ptr = vec1_contig.data_ptr<float>();
    const float* vec2_contig_ptr = vec2_contig.data_ptr<float>();
    const float* vec3_contig_ptr = vec3_contig.data_ptr<float>();
    const float* vec4_contig_ptr = vec4_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    
    scmpOnGPU <<<1, 10 >>> (vec1_contig_ptr, vec2_contig_ptr, vec3_contig_ptr, vec4_contig_ptr, result_ptr);

    cudaDeviceSynchronize();

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu", &calcOnCpu);
    m.def("gpu", &calcOnGpu);
}