import torch
import lab3
import unittest


def get_lib_result(m, n, k):
    # Gets resulting tensor from library

    X = torch.randn(m, k).cuda()
    W = torch.randn(n, k).cuda()
    b = torch.randn(n).cuda()

    return lab3.linear_layer_calc_result(X, W, b)


def get_torch_result(m, n, k):
    # Gets resulting tensor from library

    X = torch.randn(m, k).cuda()
    W = torch.randn(n, k).cuda()
    b = torch.randn(n).cuda()

    linear_layer = torch.nn.Linear(k, n).cuda()

    with torch.no_grad():
        linear_layer.weight.copy_(W)
        linear_layer.bias.copy_(b)

    return linear_layer(X)


class TestCUDALinearLayer(unittest.TestCase):
    # Test suite for linear layer

    # Test case 1
    # GPU
    def test_cuda_linear_layer_1(self):
        m = 2
        k = 2
        n = 1

        lib_res = get_lib_result(m, n, k)
        torch_res = get_torch_result(m, n, k)

        self.assertTrue(torch.allclose(lib_res, torch_res))


    # Test case 2
    # GPU
    def test_cuda_linear_layer_2(self):
        m = 4
        k = 4
        n = 2

        lib_res = get_lib_result(m, n, k)
        torch_res = get_torch_result(m, n, k)

        self.assertTrue(torch.allclose(lib_res, torch_res))


    # Test case 3
    # GPU
    def test_cuda_linear_layer_3(self):
        m = 6
        k = 6
        n = 6

        lib_res = get_lib_result(m, n, k)
        torch_res = get_torch_result(m, n, k)

        self.assertTrue(torch.allclose(lib_res, torch_res))


    # Test case 4
    # GPU
    def test_cuda_linear_layer_4(self):
        m = 10
        k = 10
        n = 5

        lib_res = get_lib_result(m, n, k)
        torch_res = get_torch_result(m, n, k)

        self.assertTrue(torch.allclose(lib_res, torch_res))


    # Test case 5
    # GPU
    def test_cuda_linear_layer_5(self):
        m = 20
        k = 20
        n = 10

        lib_res = get_lib_result(m, n, k)
        torch_res = get_torch_result(m, n, k)

        self.assertTrue(torch.allclose(lib_res, torch_res))


if __name__ == '__main__':
    unittest.main()
