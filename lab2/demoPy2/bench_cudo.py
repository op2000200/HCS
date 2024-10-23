import torch
import lab2
from simple_benchmark import benchmark
import matplotlib.pyplot as plt

def bench_CPU(i):
    array_length = i
    array_1 = torch.rand(array_length)
    array_2 = torch.rand(array_length)
    array_3 = torch.rand(array_length)
    array_4 = torch.rand(array_length)
    result_library = lab2.cpu(array_1, array_2, array_3, array_4)

def bench_GPU(i):
    array_length = i
    array_1 = torch.rand(array_length)
    array_2 = torch.rand(array_length)
    array_3 = torch.rand(array_length)
    array_4 = torch.rand(array_length)
    result_library = lab2.gpu(array_1, array_2, array_3, array_4)

funcs = [bench_CPU, bench_GPU]
arguments = {i: [i] for i in range(10_000_000,50_000_000,10_000_000)}
argument_name = 'Array Length'
aliases = {bench_CPU: 'CPU', bench_GPU: 'GPU'}
results = benchmark(funcs, arguments, argument_name, function_aliases=aliases)

results.plot()
plt.show()
