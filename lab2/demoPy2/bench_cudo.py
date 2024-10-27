import torch
import lab2
from simple_benchmark import benchmark
import matplotlib.pyplot as plt

def bench_GPU(i):
    array_length = i
    array_1 = torch.rand(array_length)
    array_2 = torch.rand(array_length)

    result_library = lab2.dot_gpu(array_1, array_2)

def bench_Dot(i):
    array_length = i
    array_1 = torch.rand(array_length)
    array_2 = torch.rand(array_length)

    result_torch = torch.dot(array_1, array_2)

funcs = [ bench_GPU,bench_Dot]
arguments = {i: [i] for i in range(10_000_000,50_000_000,10_000_000)}
argument_name = 'Array Length'
aliases = {bench_GPU: 'GPU',bench_Dot:"Dot"}
results = benchmark(funcs, arguments, argument_name, function_aliases=aliases)

results.plot()
plt.show()
