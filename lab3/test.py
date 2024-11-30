import torch
import lab3
import unittest


def get_lib_result(X, W, b):
    # Gets resulting tensor from library

    return lab3.linear_layer_calc_result(X, W, b)


def get_lib_grads(X, W, res):
    # Gets resulting grads from library

    return lab3.linear_layer_calc_grads(X, W, res)


def get_torch_result(m, n, k, X, W, b):
    # Gets resulting tensor from library

    linear_layer = torch.nn.Linear(k, n).cuda()

    with torch.no_grad():
        linear_layer.weight.copy_(W)
        linear_layer.bias.copy_(b)

    return linear_layer(X)


class TestCUDALinearLayer(unittest.TestCase):
    # Test suite for linear layer

    rtol = 1e-03
    atol = 1e-04

    # Test case 1
    # GPU
    def test_cuda_linear_layer_1(self):
        m = 2
        k = 2
        n = 1

        X = torch.randn(m, k).cuda()
        W = torch.randn(n, k).cuda()
        b = torch.randn(n).cuda()

        lib_res = get_lib_result(X, W, b)
        torch_res = get_torch_result(m, n, k, X, W, b)

        self.assertTrue(torch.allclose(lib_res, torch_res))


    # Test case 2
    # GPU
    def test_cuda_linear_layer_2(self):
        m = 4
        k = 4
        n = 2

        X = torch.randn(m, k).cuda()
        W = torch.randn(n, k).cuda()
        b = torch.randn(n).cuda()

        lib_res = get_lib_result(X, W, b)
        torch_res = get_torch_result(m, n, k, X, W, b)

        self.assertTrue(torch.allclose(lib_res, torch_res))


    # Test case 3
    # GPU
    def test_cuda_linear_layer_3(self):
        m = 6
        k = 6
        n = 6

        X = torch.randn(m, k).cuda()
        W = torch.randn(n, k).cuda()
        b = torch.randn(n).cuda()

        lib_res = get_lib_result(X, W, b)
        torch_res = get_torch_result(m, n, k, X, W, b)

        self.assertTrue(torch.allclose(lib_res, torch_res, rtol=self.rtol, atol=self.atol))


    # Test case 4
    # GPU
    def test_cuda_linear_layer_4(self):
        m = 10
        k = 10
        n = 5

        X = torch.randn(m, k).cuda()
        W = torch.randn(n, k).cuda()
        b = torch.randn(n).cuda()

        lib_res = get_lib_result(X, W, b)
        torch_res = get_torch_result(m, n, k, X, W, b)

        with torch.no_grad():
            self.assertTrue(torch.allclose(lib_res, torch_res, rtol=self.rtol, atol=self.atol))


    # Test case 5
    # GPU
    def test_cuda_linear_layer_5(self):
        m = 20
        k = 20
        n = 10

        X = torch.randn(m, k).cuda()
        W = torch.randn(n, k).cuda()
        b = torch.randn(n).cuda()

        lib_res = get_lib_result(X, W, b)
        torch_res = get_torch_result(m, n, k, X, W, b)

        self.assertTrue(torch.allclose(lib_res, torch_res))


class TestCUDALinearLayerForwardBackward(unittest.TestCase):
    # Test suite for linear layer

    rtol = 1e-03
    atol = 1e-04

    # Test case 1
    # GPU
    def test_cuda_linear_layer_forward_backward_1(self):
        m = 2
        k = 2
        n = 1

        X = torch.randn(m, k).cuda()
        W = torch.randn(n, k).cuda()
        b = torch.randn(n).cuda()

        lib_res = get_lib_result(X, W, b)
        torch_res = get_torch_result(m, n, k, X, W, b)

        # Perform backward pass with the gradient output
        print(lib_res)
        print(torch_res)
        torch_grads = torch_res.backward()

        lib_grads = get_lib_grads(X, W, lib_res)

        
        self.assertListEqual(lib_grads, torch_grads)





if __name__ == '__main__':
    unittest.main()
