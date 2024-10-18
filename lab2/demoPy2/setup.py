import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(
    name='lab2',
    version='1.0',
    ext_modules=[
        CppExtension(
            name='lab2',
            sources=['bridge.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)