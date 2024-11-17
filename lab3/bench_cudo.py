import torch
import lab3
from simple_benchmark import benchmark
import matplotlib.pyplot as plt

def bench_custom_linear(array):
    m = array[0]
    k = array[1]
    n = array[2]
    X = torch.randn(m, k).cuda()
    W = torch.randn(n, k).cuda()
    b = torch.randn(n).cuda()
    res = lab3.linear_layer_calc_result(X, W, b)

def bench_pytorch_linear(array):
    m = array[0]
    k = array[1]
    n = array[2]
    X = torch.randn(m, k).cuda()
    W = torch.randn(n, k).cuda()
    b = torch.randn(n).cuda()

    linear_layer = torch.nn.Linear(k, n).cuda()
    with torch.no_grad():
        linear_layer.weight.copy_(W)
        linear_layer.bias.copy_(b)
    res = linear_layer(X)
def all_Case_Bench(arguments, title):

    funcs = [bench_custom_linear, bench_pytorch_linear]
    argument_name = 'Sizes'
    aliases = {bench_custom_linear: f'Custom Linear Layer ({title})', bench_pytorch_linear: f'PyTorch Linear Layer ({title})'}
    results = benchmark(funcs, arguments, argument_name, function_aliases=aliases) 
    results.plot() 
    plt.title(title)

# funcs = [bench_custom_linear, bench_pytorch_linear]
# argument_name = 'Sizes'
# arguments = {(m, k, n): [m, k, n] for m in range(10, 100, 30) for k in range(10, 100, 30) for n in range(10, 100, 30)}
# arguments = {m: [m, 50, 50] for m in range(10, 100, 30)}

# aliases = {bench_custom_linear: 'Custom Linear Layer', bench_pytorch_linear: 'PyTorch Linear Layer'}
# results = benchmark(funcs, arguments, argument_name, function_aliases=aliases)

# results.plot()
range_Start = 1000
range_End = 3000
range_Step_Buf = 1000
range_Step_Main = 500

arguments1 = {m: [m, buf, buf] for buf in range(range_Start,range_End + range_Start,range_Step_Buf) for m in range(range_Start, range_End + range_Start, range_Step_Main)} 
all_Case_Bench(arguments1, 'm')

arguments2 = {k: [buf, k, buf] for buf in range(range_Start,range_End + range_Start,range_Step_Buf) for k in range(range_Start, range_End + range_Start, range_Step_Main)} 
all_Case_Bench(arguments2, 'k')

arguments3 = {n: [buf, buf, n] for buf in range(range_Start,range_End + range_Start,range_Step_Buf) for n in range(range_Start, range_End + range_Start, range_Step_Main)} 
all_Case_Bench(arguments3, 'n')

plt.show()
