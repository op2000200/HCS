from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='scmp.cu',
      ext_modules=[cpp_extension.CUDAExtension('scmp.cu', ['scmp.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
