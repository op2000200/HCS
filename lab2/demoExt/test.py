import torch
import lab2
import unittest


class TestCUDADotProduct(unittest.TestCase):

    # Test case 1
    # CPU
    def test_cpu(self):
        array_length = 10000
        array_1 = torch.rand(array_length)
        array_2 = torch.rand(array_length)

        result_library = lab2.dot_cpu(array_1, array_2)
        result_torch = torch.dot(array_1, array_2)

        self.assertTrue(torch.allclose(result_library, result_torch))


    # Test case 1
    # GPU
    def test_gpu(self):
        array_length = 10000
        array_1 = torch.rand(array_length)
        array_2 = torch.rand(array_length)

        result_library = torch.tensor([lab2.dot_gpu(array_1, array_2)])
        result_torch = torch.dot(array_1, array_2)

        self.assertTrue(torch.allclose(result_library, result_torch))


if __name__ == '__main__':
    unittest.main()
