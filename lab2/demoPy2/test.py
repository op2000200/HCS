import torch
import lab2
import unittest


class TestCUDAModule(unittest.TestCase):

    test1_check_tensor = torch.Tensor([3, 11, 23, 39, 59])
    # Test case 1
    # CPU
    def test_cpu_1(self):
        vec1 = torch.empty(5)
        vec2 = torch.empty(5)
        vec3 = torch.empty(5)
        vec4 = torch.empty(5)

        for i in range(5):
            vec1[i] = i
            vec2[i] = i + 1
            vec3[i] = i + 2
            vec4[i] = i + 3

        res = lab2.cpu(vec1,vec2,vec3,vec4)

        self.assertTrue(torch.allclose(res, self.test1_check_tensor))


    # Test case 1
    # GPU
    def test_gpu_1(self):
        vec1 = torch.empty(5)
        vec2 = torch.empty(5)
        vec3 = torch.empty(5)
        vec4 = torch.empty(5)

        for i in range(5):
            vec1[i] = i
            vec2[i] = i + 1
            vec3[i] = i + 2
            vec4[i] = i + 3

        res = lab2.gpu(vec1,vec2,vec3,vec4)
        self.assertTrue(torch.allclose(res, self.test1_check_tensor))



if __name__ == '__main__':
    unittest.main()
