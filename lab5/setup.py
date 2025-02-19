import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name='lab3',
    version='1.0',
    ext_modules=[
        CUDAExtension(
            name='lab3',
            sources=['lib.cu'],
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)