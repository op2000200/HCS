import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name='lab2',
    version='1.2',
    ext_modules=[
        CUDAExtension(
            name='lab2',
            sources=['scmp.cu'],
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)