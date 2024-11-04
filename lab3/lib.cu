#include <torch/extension.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>

const unsigned int block_size = 32;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void linear_layer(torch::PackedTensorAccessor32<float,2> X, 
                            torch::PackedTensorAccessor32<float,2> W, 
                            torch::PackedTensorAccessor32<float,1> b, 
                            torch::PackedTensorAccessor32<float,2> result)
{
    auto n = blockDim.x * blockIdx.x + threadIdx.x;
    auto m = blockDim.y * blockIdx.y + threadIdx.y;

    float acc = 0;
    if (m < X.size(0) && n < W.size(0)) 
    {
        for (int k = 0; k < X.size(1); k++) 
        {
            acc += X[m][k] * W[n][k];
        }
        result[m][n] = acc + b[n];
    }
}

__global__ void gradient_weight()
{

}

__global__ void gradient_bias()
{

}

torch::Tensor linear_layer_forward(torch::Tensor X, torch::Tensor W, torch::Tensor b)
{
    CHECK_INPUT(X); CHECK_INPUT(W); CHECK_INPUT(b);

    int b_count = b.numel();
    int X_W_row_count = W.numel() / b_count;
    int X_collumn_count = X.numel() / X_W_row_count;

    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCUDA).requires_grad(true);
    torch::Tensor result = torch::zeros({X_collumn_count,b_count},options);

    dim3 grid(2,2);
    dim3 block(block_size, block_size);
    linear_layer<<<grid,block>>>(
        X.packed_accessor32<float,2>(),
        W.packed_accessor32<float,2>(),
        b.packed_accessor32<float,1>(),
        result.packed_accessor32<float,2>()
    );
    return result;
}

std::vector<torch::Tensor> linear_layer_backward()
{
    std::vector<torch::Tensor> one;
    return one;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_layer_forward", &linear_layer_forward);
    m.def("linear_layer_backward", &linear_layer_backward);
}