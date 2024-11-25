import torch
from torch.utils import cpp_extension
import unittest
import math
import os
import torch.nn as nn
from torch.nn.functional import (
    linear as torch_linear,
    relu
)


class LinearFunction(torch.autograd.Function):
    r"""Evaluates the expression :math:`xA^T + b` and its gradient"""

    @staticmethod
    def up_backend(backend_source='lib.cu'):
        build_dir = f"./build/{backend_source.replace('/', '.')}"

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        LinearFunction.backend = cpp_extension.load(
            name='lab3',
            sources=backend_source,
            extra_cuda_cflags=[
                '-std=c++17',
                '--extended-lambda',
                '-O3'
            ],
            extra_cflags=['-O3'],
            build_directory=build_dir
        )

    @staticmethod
    def forward(ctx, input, weight, bias):
        result = LinearFunction.backend.linear_layer_calc_result(input, weight, bias)
        ctx.save_for_backward(input, weight, result)
        return result

    @staticmethod
    def backward(ctx):
        d_input, d_weight, d_bias = LinearFunction.backend.linear_layer_calc_grads(
            *ctx.saved_tensors)
        return d_input, d_weight, d_bias


class GenericTestCase(unittest.TestCase):
    @staticmethod
    def init_parameters(weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    @classmethod
    def setUpClass(cls):
        LinearFunction.up_backend(cls.backend)

    def _test_generic(self, dtype, backward):
        if (dtype in [torch.float16, torch.float64]
                and torch.cuda.get_device_capability()[0] < 7):
            self.skipTest('Unsupported CUDA device.')

        tensor_opt = {
            'device': 'cuda',
            'dtype': dtype,
            'requires_grad': backward
        }

        if self.layout_x16:
            x = torch.ones((128, 9216), **tensor_opt)
            w1 = torch.empty((4096, 9216), **tensor_opt)
            b1 = torch.empty((4096, ), **tensor_opt)
            w2 = torch.empty((16, 4096), **tensor_opt)
            b2 = torch.empty((16, ), **tensor_opt)
        else:
            x = torch.ones((127, 9215), **tensor_opt)
            w1 = torch.empty((4095, 9215), **tensor_opt)
            b1 = torch.empty((4095, ), **tensor_opt)
            w2 = torch.empty((15, 4095), **tensor_opt)
            b2 = torch.empty((15, ), **tensor_opt)

        GenericTestCase.init_parameters(w1, b1)
        GenericTestCase.init_parameters(w2, b2)

        y = relu(LinearFunction.apply(x, w1, b1), inplace=True)
        z = relu(LinearFunction.apply(y, w2, b2), inplace=True)

        x_ = x.detach().clone().requires_grad_()
        w1_ = w1.detach().clone().requires_grad_()
        b1_ = b1.detach().clone().requires_grad_()
        w2_ = w2.detach().clone().requires_grad_()
        b2_ = b2.detach().clone().requires_grad_()

        y_ = relu(torch_linear(x_, w1_, b1_), inplace=True)
        z_ = relu(torch_linear(y_, w2_, b2_), inplace=True)

        match dtype:
            case torch.float16:
                tol = {'atol': 1e-2, 'rtol': 1e-1}
            case torch.float32:
                tol = {'atol': 1e-4, 'rtol': 1e-3}
            case torch.float64:
                tol = {'atol': 1e-9, 'rtol': 1e-8}

        def max_diff(a, b):
            return (a - b).abs().max().item()

        with torch.no_grad():
            self.assertTrue(
                torch.allclose(z, z_, **tol),
                f'max diff (z, z_): {max_diff(z, z_)}'
            )

        if not backward:
            return

        z_.backward(torch.ones_like(z_))   #
        z.backward(torch.ones_like(z))    #


        with torch.no_grad():
            self.assertTrue(
                torch.allclose(x.grad, x_.grad, **tol),
                f'max diff (x.grad, x_.grad): {max_diff(x.grad, x_.grad)}'
            )
            self.assertTrue(
                torch.allclose(w1.grad, w1_.grad, **tol),
                f'max diff (w1.grad, w1_.grad): {max_diff(w1.grad, w1_.grad)}'
            )
            self.assertTrue(
                torch.allclose(b1.grad, b1_.grad, **tol),
                f'max diff (b1.grad, b1_.grad): {max_diff(b1.grad, b1_.grad)}'
            )
            self.assertTrue(
                torch.allclose(w2.grad, w2_.grad, **tol),
                f'max diff (w2.grad, w2_.grad): {max_diff(w2.grad, w2_.grad)}'
            )
            self.assertTrue(
                torch.allclose(b2.grad, b2_.grad, **tol),
                f'max diff (b2.grad, b2_.grad): {max_diff(b2.grad, b2_.grad)}'
            )


class TestCaseFactory(type):
    def __new__(cls, name, base, attrs, **kwargs):
        assert GenericTestCase in base
        attrs.update(kwargs)
        TestCaseFactory.__add_tests(attrs, **kwargs)
        return super().__new__(cls, name, base, attrs)

    @staticmethod
    def __add_test(attrs, backend, dtype, layout_x16, backward):
        method_name = TestCaseFactory.__generate_test_name(
            backend, dtype, layout_x16, backward
        )
        attrs[method_name] = \
            (lambda self, d=dtype, b=backward:
                GenericTestCase._test_generic(self, d, b))

    @staticmethod
    def __add_tests(attrs, backend, dtypes, layout_x16, backward):
        for dtype in dtypes:
            TestCaseFactory.__add_test(
                attrs, backend, dtype, layout_x16, backward)

    @staticmethod
    def __generate_test_name(backend, dtype, layout_x16, backward):
        dtype_lb = str(dtype).split('.')[-1]
        backend_lb = backend.split('/')[-1].replace('.', '_')
        bkwd_lb = 'forward_backward' if backward else 'forward'

        if layout_x16:
            return f'test_{backend_lb}_{dtype_lb}_layout_x16_{bkwd_lb}'
        else:
            return f'test_{backend_lb}_{dtype_lb}_{bkwd_lb}'


class Lab3TestCaseGrid2d(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float32],
    backward=True, backend='lib.cu', layout_x16=True
):

    pass


class Lab3TestCaseGrid2dBadLayout(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float32],
    backward=True, backend='lib.cu', layout_x16=False
):

    pass


class Lab3TestCaseGrid3d(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float32],
    backward=True, backend='lib.cu', layout_x16=True
):

    pass


class Lab3TestCaseGrid3dBadLayout(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float32],
    backward=True, backend='lib.cu', layout_x16=False
):

    pass


if __name__ == '__main__':
    unittest.main()
