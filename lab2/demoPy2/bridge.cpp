#include <torch/extension.h>
//#include "scmp.cu"

int checkk(int in)
{
    return in;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check", &checkk);
}