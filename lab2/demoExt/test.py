import torch
import lab3
import unittest


class TestCUDALinearLayer(unittest.TestCase):


    # Test case 1
    # GPU
    def test_gpu(self):
        m = 2
        k = 2
        n = 1

        X = torch.randn(m, k).cuda()
        W = torch.randn(n, k).cuda()
        b = torch.randn(n).cuda()

        lib_res = lab3.linear_layer_calc_result(X, W, b)

        linear_layer = torch.nn.Linear(k, n).cuda()

        with torch.no_grad():
            linear_layer.weight.copy_(W)
            linear_layer.bias.copy_(b)

        torch_res = linear_layer(X)

        self.assertTrue(torch.allclose(lib_res, torch_res))


if __name__ == '__main__':
    unittest.main()
