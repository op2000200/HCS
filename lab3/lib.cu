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

    float buf = 0;
    if (m < X.size(0) && n < W.size(0)) 
    {
        for (int k = 0; k < X.size(1); k++) 
        {
            buf += X[m][k] * W[n][k];
        }
        result[m][n] = buf + b[n];
    }
}

__global__ void gradient_input(torch::PackedTensorAccessor32<float,2> resultInput,
                               torch::PackedTensorAccessor32<float,2> W,
                               torch::PackedTensorAccessor32<float,2> gradInputHolder)
{
    auto n = blockDim.x * blockIdx.x + threadIdx.x;
    auto m = blockDim.y * blockIdx.y + threadIdx.y;

    float buf = 0;
    if (m < resultInput.size(0) && n < W.size(1)) {
        for (int k = 0; k < resultInput.size(1); k++) {
            buf += resultInput[m][k] * W[k][n];
        }
        gradInputHolder[m][n] = buf;
    }
}

__global__ void gradient_weight(torch::PackedTensorAccessor32<float,2> resultInput,
                                torch::PackedTensorAccessor32<float,2> Y,
                                torch::PackedTensorAccessor32<float,2> gradWeightHolder)
{
    auto n = blockDim.x * blockIdx.x + threadIdx.x;
    auto m = blockDim.y * blockIdx.y + threadIdx.y;
 
    if (m < resultInput.size(1) && n < Y.size(1)) {
        float buf = 0;
        for (int k = 0; k < resultInput.size(0); k++) {
            buf += resultInput[k][m] * Y[k][n];
        }
        gradWeightHolder[m][n] = buf;
    }
}

__global__ void gradient_bias(torch::PackedTensorAccessor32<float,2> resultInput,
                              torch::PackedTensorAccessor32<float,1> gradBiasHolder)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int n = resultInput.size(0);
    int m = resultInput.size(1);

    if (i < m) {
        float buf = 0;
        for (int j = 0; j < n; j++) {
            buf += resultInput[j][i];
        }
        gradBiasHolder[i] = buf;
    }
}

int calc_grid_size(int m) {
    return (m + block_size - 1) / block_size;
}

torch::Tensor linear_layer_calc_result(torch::Tensor X, torch::Tensor W, torch::Tensor b)
{
    CHECK_INPUT(X); CHECK_INPUT(W); CHECK_INPUT(b);

    int b_count = b.numel();
    int X_W_row_count = W.numel() / b_count;
    int X_collumn_count = X.numel() / X_W_row_count;

    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCUDA).requires_grad(true);
    torch::Tensor result = torch::zeros({X_collumn_count,b_count},options);

    dim3 grid(calc_grid_size(X_W_row_count), calc_grid_size(X_collumn_count)); // TODO: fix the shit  UPD: fixed
    dim3 block(block_size, block_size);
    linear_layer<<<grid,block>>>(
        X.packed_accessor32<float,2>(),
        W.packed_accessor32<float,2>(),
        b.packed_accessor32<float,1>(),
        result.packed_accessor32<float,2>()
    );
    cudaDeviceSynchronize();
    return result;
}

std::vector<torch::Tensor> linear_layer_calc_grads(torch::Tensor X, torch::Tensor W, torch::Tensor result)
{
    CHECK_INPUT(X); CHECK_INPUT(W); CHECK_INPUT(result);

    auto x = X.packed_accessor32<float, 2>();
    auto w = W.packed_accessor32<float, 2>();
    auto res = result.packed_accessor32<float, 2>();
    

    int m = x.size(0);
    int n = res.size(1);
    int k = x.size(1);

    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCUDA).requires_grad(true);

    torch::Tensor gradientInput = torch::zeros({m, k}, options);
    torch::Tensor gradientWeight = torch::zeros({n, k}, options);
    torch::Tensor gradientBias = torch::zeros({n, }, options);

    dim3 grid(calc_grid_size(n), calc_grid_size(m)); // TODO: fix the shit  UPD: fixed
    dim3 block(block_size, block_size);

    gradient_input<<<grid, block>>>(
        res,
        w,
        gradientInput.packed_accessor32<float, 2>()
    );

    dim3 grid(calc_grid_size(k), calc_grid_size(n)); // TODO: fix the shit  UPD: fixed
    gradient_weight<<<grid, block>>>(
        res,
        x,
        gradientWeight.packed_accessor32<float, 2>()
    );

    gradient_bias<<<calc_grid_size(n), block>>>(
        res,
        gradientBias.packed_accessor32<float, 1>()
    );

    return std::vector<torch::Tensor>{gradientInput, gradientWeight, gradientBias};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_layer_calc_result", &linear_layer_calc_result);
    m.def("linear_layer_calc_grads", &linear_layer_calc_grads);
}
